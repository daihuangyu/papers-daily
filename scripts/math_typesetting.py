"""MathJax assets and ReportLab markup for native Markdown math delimiters."""
import hashlib
import html
import json
import re
import subprocess
import inspect
from pathlib import Path
from reportlab.platypus import Image, Spacer, KeepTogether

INLINE = re.compile(r'(?<!\\)\$([^$\n]+?)(?<!\\)\$')


def enable_cjk_inline_images():
    """Guard empty image fragments in ReportLab's CJK word-break heuristic.

    This is an in-process compatibility fix; no installed library is edited.
    An image fragment has width but empty text, so ord(fragment) is invalid.
    """
    import reportlab.platypus.paragraph as paragraph
    source = inspect.getsource(paragraph.cjkFragSplit)
    if 'if ord(u)<0x3000:' not in source:
        return
    source = source.replace('if ord(u)<0x3000:', 'if u and ord(u)<0x3000:')
    source = source.replace("if uj and category(uj)=='Zs' or ord(uj)>=0x3000:",
                            "if uj and (category(uj)=='Zs' or ord(uj)>=0x3000):")
    # '' is a substring of every punctuation string, but an empty image
    # fragment is not punctuation and must be moved to the next line.
    source = source.replace('if u not in ALL_CANNOT_START and i>lineStartPos+1:',
                            'if (not u or u not in ALL_CANNOT_START) and i>lineStartPos+1:')
    exec(compile(source, '<papers-daily-cjk-image-compat>', 'exec'), paragraph.__dict__)


class MathTypesetter:
    def __init__(self, markdown):
        self.folder = markdown.parent / 'assets' / 'math'
        self.items = {}
        text = markdown.read_text(encoding='utf-8')
        # Code fences are literal; a currency dollar is not a math delimiter.
        plain = re.sub(r'```.*?```', '', text, flags=re.S)
        requests = []
        for m in re.finditer(r'^\$\$\s*\n(.*?)\n\$\$$', plain, re.M | re.S):
            requests.append(dict(tex=m[1].strip(), display=True))
        plain = re.sub(r'^\$\$\s*\n.*?\n\$\$$', '', plain, flags=re.M | re.S)
        requests += [dict(tex=m[1], display=False) for m in INLINE.finditer(plain)]
        unique = list({(r['tex'], r['display']): r for r in requests}.values())
        if not unique:
            return
        enable_cjk_inline_images()
        self.folder.mkdir(parents=True, exist_ok=True)
        request = self.folder / 'requests.json'
        request.write_text(json.dumps(unique, ensure_ascii=False), encoding='utf-8')
        subprocess.run(['node', str(Path(__file__).with_name('render_math.mjs')),
                        str(request), str(self.folder)], check=True)
        for item in json.loads((self.folder / 'index.json').read_text()):
            self.items[item['tex'], item['display']] = item
        expected = {'requests.json', 'index.json'} | {i[k] for i in self.items.values() for k in ('png', 'svg')}
        for file in self.folder.iterdir():
            if file.is_file() and file.name not in expected:
                file.unlink()

    def inline(self, text, render_prose):
        result = []; end = 0
        for match in INLINE.finditer(text):
            result.append(render_prose(text[end:match.start()]))
            item = self.items[match[1], False]
            # 16 CSS pixels == 12 typographic points; prose is 10pt.
            scale = 10 / 16
            src = html.escape(str(self.folder / item['png']), quote=True)
            result.append(f'<img src="{src}" width="{item["width"]*scale:.3f}" '
                          f'height="{item["height"]*scale:.3f}" valign="{-item["depth"]*scale:.3f}"/>')
            end = match.end()
        result.append(render_prose(text[end:]))
        return ''.join(result)

    def block(self, tex):
        item = self.items[tex.strip(), True]
        scale = min(12 / 16, 470 / item['width'])
        picture = Image(str(self.folder / item['png']), width=item['width']*scale,
                        height=item['height']*scale)
        picture.hAlign = 'CENTER'
        return KeepTogether([Spacer(1, 7), picture, Spacer(1, 12)])
