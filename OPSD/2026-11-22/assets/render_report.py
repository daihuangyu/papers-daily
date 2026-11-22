"""Rebuild this complete report with the checked layout, using the repo renderer.

Optional: set PAPERS_DAILY_HOST_FONT to an embeddable TrueType CJK font.
Without it, use the repository's normal Linux font preparation pipeline.
"""
from pathlib import Path
import os
import sys
import types
from reportlab.pdfbase.ttfonts import TTFont

folder = Path(__file__).resolve().parent.parent
repo = folder.parent.parent
markdown = folder / 'report.md'
override = os.environ.get('PAPERS_DAILY_HOST_FONT')
sys.path.insert(0, str(repo / 'scripts'))
if override:
    font = Path(override)
    face = TTFont('glyph_check', str(font))
    missing = sorted({c for c in markdown.read_text() if not c.isspace()
                      and ord(c) not in face.face.charToGlyph})
    if missing:
        raise ValueError(f'Missing glyphs: {missing}')
    font_module = types.ModuleType('cjk_font')
    font_module.embedded_font = lambda text: font
    sys.modules['cjk_font'] = font_module

renderer = repo / 'scripts/build_report.py'
source = renderer.read_text()
needle = "(440 if direction == 'CCF_Papers' else 540)"
if needle not in source:
    raise RuntimeError('Shared renderer changed; review the local layout adapter.')
source = source.replace(needle, "(430 if path.name == 'sa-table2.png' else 380)")
source = source.replace('fontSize=10,leading=17,spaceAfter=8',
                        'fontSize=10,leading=16,spaceAfter=6')
source = source.replace('story.extend([table,Spacer(1,10)]); continue',
                        'story.append(KeepTogether([table,Spacer(1,10)])); continue')
source = source.replace("max_width = widths.get(path.name, 487) if direction == 'CCF_Papers' else 487",
                        "max_width = 330 if path.name in ('rds-fig4.png', 'rds-fig6.png', 'esc-fig3.png', 'p2-table4.png', 'p2-table5.png', 'sa-fig3.png', 'sa-fig5.png') else 487")
namespace = {'__file__': str(renderer), '__name__': 'report_renderer'}
exec(compile(source, str(renderer), 'exec'), namespace)
namespace['build'](markdown)
