"""Validate source uniqueness and rebuild root README from per-day manifests."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def update():
    manifests=[]; seen={}; urls={}
    for path in sorted(ROOT.glob('*/*/manifest.json'),reverse=True):
        item=json.loads(path.read_text()); folder=path.parent.relative_to(ROOT)
        for source in item['sources']:
            sid=source['id']; url=source['url'].rstrip('/')
            if sid in seen or url in urls:
                raise ValueError(f'Duplicate source {sid}: {path}')
            seen[sid]=str(folder); urls[url]=str(folder)
        for f in ('report.md','report.pdf'):
            if not (path.parent/f).is_file(): raise ValueError(f'Missing {folder/f}')
        manifests.append((item,folder))
    lines=['# Papers Daily','', '多方向论文与硬核技术资料精读。每日报告包含机制、公式、实现细节、原图、实验条件与分级延伸阅读。','',
        '## 更新安排','', '每天北京时间 09:00，由 Codex 当前聊天的定时任务检索、写作、生成 PDF 并提交。',
        'README 由清单自动生成；同一来源 ID 或 URL 不允许重复。近期窗口内资料不足时报告实际数量，不用旧资料凑数。','',
        '## 方向','', '- [VLM](VLM/README.md)：小型视觉语言模型的训练与后训练，重点为OPD/蒸馏、RL、视觉推理和搜索agent；论文优先。首日vLLM工程资料作为历史内容保留。','',
        '## 每日索引','', '| 日期 | 方向 / 子主题 | 数量 | 阅读 |','| --- | --- | --- | --- |']
    for item,folder in manifests:
        lines.append(f"| {item['date']} | {item['direction']} / {item['topic']} | {len(item['sources'])} | [Markdown]({folder}/report.md) · [PDF]({folder}/report.pdf) |")
    lines += ['', '## 条目索引','']
    for item,folder in manifests:
        lines.extend([f"### {item['date']} · {item['topic']}",''])
        for n,s in enumerate(item['sources'],1): lines.append(f"{n}. [{s['title']}]({s['url']}) · {s['published']} · [精读]({folder}/report.md)")
    lines += ['', '## 仓库约定','', '`方向/YYYY-MM-DD/` 内含 report.md、report.pdf、manifest.json 与 assets/；scripts/ 管理 PDF、索引和发布。',
        '基础教程可早于近期窗口，但不算入每日5篇。图片逐图注明出处，原图版权归原作者；教学图和推导明确标注。',
        '每日发布流程见 [AUTOMATION.md](AUTOMATION.md)。','']
    (ROOT/'README.md').write_text('\n'.join(lines))
    (ROOT/'sources.json').write_text(json.dumps(seen,ensure_ascii=False,indent=2)+'\n')
    print(f'Validated {len(manifests)} reports and {len(seen)} unique sources')

if __name__=='__main__': update()
