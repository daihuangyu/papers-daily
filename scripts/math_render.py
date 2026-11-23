"""Shared GitHub-LaTeX -> vector display / high-DPI inline PDF renderer."""
import hashlib
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path
from reportlab.platypus import Paragraph

TOKENS = re.compile(r'(?P<ticked>(?<!\\)\$`[^`\n]+`\$)|(?P<code>`[^`\n]*`)|(?P<plain>(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$(?!\$))')


class MathParagraph(Paragraph):
    """Give inline objects a Unicode replacement char for CJK line splitting.

    ReportLab's CJK splitter calls ord() on inline image fragments; its default
    empty text raises TypeError when the image falls at a line boundary. U+FFFC
    represents an inline object, takes the image's width, and is never painted.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for frag in self.frags:
            if getattr(getattr(frag,'cbDefn',None),'kind',None)=='img' and not frag.text:
                frag.text='\ufffc'


def math_segments(text):
    """Yield (is_math, text/TeX), preserving literal code and GitHub $`...`$."""
    last=0
    for match in TOKENS.finditer(text):
        if match.group('code'):
            continue
        yield False,text[last:match.start()]
        raw=match.group()
        yield True,raw[2:-2] if match.group('ticked') else raw[1:-1]
        last=match.end()
    yield False,text[last:]


def inline_math(text):
    return (part for is_math,part in math_segments(text) if is_math)


def math_blocks(lines):
    """Yield (start, end-exclusive, TeX); ignore ordinary fenced code."""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('```'):
            start, is_math = i, line == '```math'
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                i += 1
            if is_math:
                if i == len(lines):
                    raise ValueError('Unclosed math fence')
                yield start, i + 1, '\n'.join(lines[start+1:i])
            i += 1
        elif line.startswith('$$'):
            start = i
            if line != '$$' and line.endswith('$$'):
                yield i, i + 1, line[2:-2]
                i += 1
            else:
                if line != '$$':
                    raise ValueError('Use $$ on its own line or $$expression$$')
                i += 1
                while i < len(lines) and lines[i].strip() != '$$':
                    i += 1
                if i == len(lines):
                    raise ValueError('Unclosed display math')
                yield start, i + 1, '\n'.join(lines[start+1:i])
                i += 1
        else:
            i += 1


class MathRenderer:
    def __init__(self, markdown):
        self.markdown = Path(markdown)
        lines = self.markdown.read_text(encoding='utf-8').splitlines()
        self.blocks = {a: (b, tex) for a, b, tex in math_blocks(lines)}
        excluded = set()
        code = False
        expressions = set()
        for start, (end, tex) in self.blocks.items():
            excluded.update(range(start, end))
            expressions.add((tex, True))
        for i, line in enumerate(lines):
            if i in excluded:
                continue
            if line.strip().startswith('```'):
                code = not code
            elif not code:
                expressions.update((tex, False) for tex in inline_math(line))
        self.directory = Path(tempfile.gettempdir())/'papers-daily-math'
        self.entries = {}
        self.used = []
        if not expressions:
            return
        requests = []
        for tex, display in sorted(expressions):
            if re.search(r'\\operatorname\b',tex):
                raise ValueError('GitHub rejects \\operatorname; use \\mathrm for named functions')
            identity = hashlib.sha256(('mathjax-3.2.2-v1\0'+str(display)+'\0'+tex).encode()).hexdigest()[:24]
            requests.append(dict(id=identity,tex=tex,display=display,size=12 if display else 10))
        response = subprocess.run(['node',str(Path(__file__).with_name('render_math.cjs'))],
            input=json.dumps(dict(directory=str(self.directory),expressions=requests)),
            text=True,capture_output=True)
        if response.returncode:
            raise RuntimeError('Math rendering failed (install npm dependencies with npm ci): '+response.stderr.strip())
        for item in json.loads(response.stdout):
            self.entries[item['tex'],item['display']] = item

    def display(self, tex, width=487):
        from svglib.svglib import svg2rlg
        item = self.entries[tex,True]
        drawing = svg2rlg(str(self.directory/(item['id']+'.svg')))
        if drawing is None:
            raise ValueError('Could not read rendered formula SVG')
        scale = min(1,width/drawing.width)
        if scale < 0.75:
            raise ValueError('Formula is too wide; split it using aligned: '+tex)
        drawing.scale(scale,scale)
        drawing.width *= scale
        drawing.height *= scale
        drawing.hAlign = 'CENTER'
        self.used.append((tex,True))
        return drawing

    def inline(self, text, plain_formatter, font_size=10):
        output=[]
        for is_math,part in math_segments(text):
            if not is_math:
                output.append(plain_formatter(part));continue
            item=self.entries[part,False]
            scale=font_size/item['size']
            image=self.directory/(item['id']+'.png')
            output.append('<img src="{}" width="{:.4f}" height="{:.4f}" valign="{:.4f}"/>'.format(
                html.escape(str(image),quote=True),item['width']*scale,item['height']*scale,-item['depth']*scale))
            self.used.append((part,False))
        return ''.join(output)

    def write_manifest(self):
        if not self.entries:
            return
        unused = set(self.entries)-set(self.used)
        if unused:
            raise ValueError('Formulae were parsed but not rendered: '+repr(unused))
        output = []
        for (tex,display),item in self.entries.items():
            svg = self.directory/(item['id']+'.svg')
            png = self.directory/(item['id']+'.png')
            output.append(dict(tex=tex,display=display,width_pt=item['width'],height_pt=item['height'],
                rendered_svg_sha256=hashlib.sha256(svg.read_bytes()).hexdigest(),
                inline_png_sha256=None if display else hashlib.sha256(png.read_bytes()).hexdigest()))
        target=self.markdown.parent/'assets/math-manifest.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(dict(engine='MathJax 3.2.2 SVG / svglib',
            display='vector paths',inline='transparent 576-DPI PNG, baseline aligned',
            source_sha256=hashlib.sha256(self.markdown.read_bytes()).hexdigest(),
            rendered_occurrences=len(self.used),expressions=output),ensure_ascii=False,indent=2)+'\n')
