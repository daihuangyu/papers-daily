"""Validate cross-day aliases and rebuild all direction indexes from manifests."""
import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
ARXIV = re.compile(r'(?:arxiv:|(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf|html)/)?((?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7}))(?:v\d+)?(?:\.pdf)?$', re.I)
TRACKING = {'fbclid', 'gclid', 'ref', 'source'}


def normalized_title(title):
    return ''.join(c for c in unicodedata.normalize('NFKC', title).casefold() if c.isalnum())


def arxiv_id(value):
    match = ARXIV.fullmatch(str(value).strip().rstrip('/').split('?')[0].split('#')[0])
    return match[1].lower() if match else None


def doi_id(value):
    value = unquote(str(value).strip()).casefold()
    if value.startswith(('http://', 'https://')):
        value = value.split('?')[0].split('#')[0]
    value = re.sub(r'^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)', '', value).rstrip('/')
    return value if re.fullmatch(r'10\.\d{4,9}/\S+', value) else None


def canonical_url(value):
    value = str(value).strip()
    aid, doi = arxiv_id(value), doi_id(value)
    if aid:
        return 'https://arxiv.org/abs/' + aid
    if doi:
        return 'https://doi.org/' + doi
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError(f'Expected absolute source URL: {value}')
    query = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                   if not k.casefold().startswith('utm_') and k.casefold() not in TRACKING)
    return urlunsplit(('https', parsed.netloc.casefold(), parsed.path.rstrip('/'), urlencode(query), ''))


def aliases(source):
    result = {'id:' + source['id'].strip().casefold(), 'title:' + normalized_title(source['title'])}
    result.update('title:' + normalized_title(t) for t in source.get('title_aliases', []) if t.strip())
    values = [source['id'], source['url'], source.get('arxiv_id', ''), source.get('doi', '')]
    values += source.get('canonical_urls', [])
    for value in values:
        if not value:
            continue
        aid, doi = arxiv_id(value), doi_id(value)
        if aid:
            result.add('arxiv:' + aid)
        if doi:
            result.add('doi:' + doi)
        if str(value).startswith(('http://', 'https://')):
            result.add('url:' + canonical_url(value))
    return sorted(result)


def validate_manifest(item, path):
    day = date.fromisoformat(item['date'])
    if path.parent.name != item['date'] or path.parent.parent.name != item['direction']:
        raise ValueError(f'Manifest date/direction does not match directory: {path}')
    sources = item['sources']
    if len(sources) > 5:
        raise ValueError(f'At most 5 primary papers: {path}')
    if item['direction'] == 'CCF_Papers':
        if item.get('selection_year') != day.year or item.get('selection_basis') != 'acceptance_year':
            raise ValueError(f'CCF selection must use the current acceptance year: {path}')
        if len(sources) < 5 and not item.get('shortfall_reason'):
            raise ValueError(f'Report the actual shortfall reason: {path}')
        for source in sources:
            if source.get('type') != 'paper' or source.get('paper_type') not in ('main_conference_full_regular', 'journal_regular'):
                raise ValueError(f'CCF item must be a main/journal full/regular paper: {source["id"]}')
            if source.get('acceptance_year') != day.year or source.get('ccf_rank') not in ('A', 'B'):
                raise ValueError(f'CCF rank/acceptance year is invalid: {source["id"]}')
            if source.get('presentation_type') not in ('oral', 'poster', 'spotlight', 'unknown'):
                raise ValueError(f'Invalid presentation type: {source["id"]}')
            evidence = source.get('official_evidence', {})
            for key in ('ccf_classification', 'acceptance'):
                if not evidence.get(key, {}).get('verified') or not evidence[key].get('url'):
                    raise ValueError(f'Missing verified official {key} evidence: {source["id"]}')
            if source['presentation_type'] != 'unknown':
                if not evidence.get('presentation', {}).get('verified') or not evidence['presentation'].get('url'):
                    raise ValueError(f'Missing verified official presentation evidence: {source["id"]}')
            citation = source.get('citation_count', {})
            count = citation.get('value')
            if count is not None and (type(count) is not int or count < 0 or not citation.get('source_url') or not citation.get('queried_at')):
                raise ValueError(f'Citation count needs source and query date: {source["id"]}')
            if count is None and not citation.get('unavailable_reason'):
                raise ValueError(f'Unknown citation count needs a reason: {source["id"]}')
    if item['direction'] == 'VLM' and item['date'] != '2026-11-16':
        # A calendar month, clamping the day at the previous month's end.
        import calendar
        year, month = (day.year - 1, 12) if day.month == 1 else (day.year, day.month - 1)
        start = date(year, month, min(day.day, calendar.monthrange(year, month)[1]))
        if item.get('window_start') != start.isoformat() or item.get('window_end') != day.isoformat():
            raise ValueError(f'Window must be previous calendar month through run date: {path}')
        if len(sources) < 5 and not item.get('shortfall_reason'):
            raise ValueError(f'Report the actual shortfall reason: {path}')
        for source in sources:
            if source.get('type') != 'paper':
                raise ValueError(f'Primary item must be a paper: {source["id"]}')
            if not start <= date.fromisoformat(source['published']) <= day:
                raise ValueError(f'Paper first-publication date outside window: {source["id"]}')
            if not any(a.startswith(('arxiv:', 'doi:')) for a in aliases(source)) and not source.get('publication_evidence_url'):
                raise ValueError(f'Paper needs arXiv/DOI or primary publication evidence: {source["id"]}')
    return item


def direction_intro(direction):
    if direction == 'CCF_Papers':
        return ['# CCF-A/B 年度录用论文精读', '',
            '覆盖AI/LLM训练、推理、对齐、采样与相关基础研究；按当年会议/期刊录用年度筛选，预印本可以更早公开，不采用最近一个月窗口。',
            '目标每天5篇，偏好4篇CCF-A加1篇CCF-B、3篇oral加2篇poster；spotlight独立记录，官方证据优先于配比。只纳入主会full/regular或期刊regular论文。',
            '引用数保留来源与查询日，跨所有方向和日期按论文ID、DOI、URL、标题及改题别名去重。',
            '计划北京时间每天10:00；运行和发布约定见 [AUTOMATION.md](AUTOMATION.md)，日程启用状态由调度端核验。', '']
    if direction == 'OPSD':
        return ['# OPD & OPSD 方向', '',
            '聚焦 On-Policy Distillation 与 On-Policy Self-Distillation：特权信息、监督选择、token级credit、训练稳定性及系统效率。',
            '北京时间每天09:15运行；最近一个日历月目标5篇新论文，跨方向查重，不足如实报告。',
            '原文事实、本文推导和缺失信息分别标注；每日报告含全文PDF、原始图表及可复现性检查。', '']
    if direction == 'VLM':
        return ['# VLM 方向', '',
            '重点收录小型视觉语言模型（VLM/MLLM）的训练与后训练论文：SFT、RL/GRPO、偏好优化、on-policy蒸馏、细粒度视觉感知、视觉推理和多模态搜索/工具agent。',
            '参考方向包括[Vision-OPD](https://arxiv.org/abs/2605.18740)与[MMSearch-R1](https://arxiv.org/abs/2506.20670)，作为方向锚点，不计入近期新论文。', '',
            '目标每日报告5篇；按首次公开日期筛选最近一个日历月，跨日期与方向去重，不足如实说明。官方/作者博客和基础教程仅作辅助。',
            '2026-11-16的5篇vLLM工程文章作为用户确认的首日例外保留。后续主题为VLM训练与后训练。', '',
            '运行方式与迁移验证见 [AUTOMATION.md](../AUTOMATION.md)。', '']
    return [f'# {direction} 方向', '']


def update(root=ROOT, remote_files=None):
    remote_files = remote_files or set()
    manifests, seen, records = [], {}, []
    for path in sorted(root.glob('*/*/manifest.json'), reverse=True):
        item = validate_manifest(json.loads(path.read_text(encoding='utf-8')), path)
        folder = path.parent.relative_to(root)
        for source in item['sources']:
            keys = aliases(source)
            for key in keys:
                if key in seen:
                    raise ValueError(f'Duplicate {key}: {path}; already in {seen[key]}')
            for key in keys:
                seen[key] = str(folder)
            records.append({**source, 'folder': str(folder), 'aliases': keys})
        for filename in ('report.md', 'report.pdf'):
            if not (path.parent / filename).is_file() and str(folder / filename) not in remote_files:
                raise ValueError(f'Missing {folder / filename}')
        manifests.append((item, folder))
    directions = sorted({item['direction'] for item, _ in manifests})
    lines = ['# Papers Daily', '', '多方向论文精读：机制、公式逐符号解释与推导、手算直觉、实现、原图、实验条件、局限与复现。', '',
        '## 更新安排', '', '计划北京时间（Asia/Shanghai）每天09:00，由ChatGPT云端定时任务检索、写作、生成PDF并通过GitHub连接普通提交到main。任务启用与环境验证状态见 [AUTOMATION.md](AUTOMATION.md)。',
        '目标为每次从仓库读取状态、不依赖个人电脑。云端环境绑定尚待核验，托管日程暂停；迁移期间旧本地日程补位，仍依赖电脑开机。当前状态以AUTOMATION.md为准。',
        'OPD & OPSD专项在北京时间每天09:15由既有本地日程执行，归档到OPSD/；上述09:00为VLM计划。',
        'VLM与OPSD按最近一个日历月筛选；CCF_Papers按当年录用年度筛选，计划北京时间每天10:00，不限制预印本首次公开月份。',
        '每方向目标5篇；按arXiv去版本ID、DOI、规范URL、规范标题及改题别名跨日期与方向去重，不足如实报告，不用博客凑数。', '', '## 方向', '']
    lines += [f'- [{d}]({d}/README.md)' for d in directions]
    lines += ['', '## 每日索引', '', '| 日期 | 方向 / 子主题 | 数量 | 阅读 |', '| --- | --- | --- | --- |']
    for item, folder in manifests:
        lines.append(f'| {item["date"]} | {item["direction"]} / {item["topic"]} | {len(item["sources"])} | [Markdown]({folder}/report.md) · [PDF]({folder}/report.pdf) |')
    lines += ['', '## 条目索引', '']
    for item, folder in manifests:
        lines += [f'### {item["date"]} · {item["topic"]}', '']
        lines += [f'{n}. [{s["title"]}]({s["url"]}) · {s["published"]} · [精读]({folder}/report.md)' for n, s in enumerate(item['sources'], 1)]
    lines += ['', '## 仓库约定', '', '`方向/YYYY-MM-DD/` 内含report.md、同内容report.pdf、manifest.json与assets/；其他方向使用独立一级目录。',
        'PDF嵌入中文字体，实际执行环境与字体记录在当日assets/QA.md；Linux云端默认使用Noto。基础教程可早于近期窗口，但不计入每日5篇。原图注明论文图号、来源并逐图解释。',
        '公式采用LaTeX，GitHub原生渲染，PDF同步排版；[公式语法与构建说明](scripts/MATH.md)。',
        'README与各方向索引及sources.json由scripts/update_index.py根据全部manifest重建。sources.json v2保存每个来源的全部去重别名。',
        '每日发布和失败处理见 [AUTOMATION.md](AUTOMATION.md)。', '']
    (root / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    for direction in directions:
        intro = direction_intro(direction)
        for item, folder in manifests:
            if item['direction'] == direction:
                intro += [f'- [{item["date"]}：{item["topic"]}]({item["date"]}/report.md) · [PDF]({item["date"]}/report.pdf) · {len(item["sources"])}篇']
        (root / direction / 'README.md').write_text('\n'.join(intro) + '\n', encoding='utf-8')
    (root / 'sources.json').write_text(json.dumps({'schema_version': 2, 'sources': records, 'aliases': seen}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Validated {len(manifests)} reports, {len(records)} unique sources, {len(seen)} aliases')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote-tree', type=Path, help='GitHub recursive tree JSON for unchanged remote files in a sparse cloud workspace')
    args = parser.parse_args()
    remote = set()
    if args.remote_tree:
        inventory = json.loads(args.remote_tree.read_text(encoding='utf-8'))
        if inventory.get('truncated'):
            raise ValueError('Remote inventory is truncated; fetch the complete tree')
        remote = {e['path'] for e in inventory['tree'] if e['type'] == 'blob' and e.get('sha') and e.get('size', 0) > 0}
    update(remote_files=remote)
