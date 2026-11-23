"""Regression checks for math embedded in Chinese ReportLab paragraphs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('math_typesetting', Path(__file__).resolve().parents[1] / 'math_typesetting.py')
math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math)


class MathLayoutTests(unittest.TestCase):
    def test_cjk_image_wraps_and_expands_line_height(self):
        from PIL import Image
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph
        math.enable_cjk_inline_images()
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'formula.png'
            Image.new('RGB', (200, 100), 'white').save(image)
            style = ParagraphStyle('cjk-math', fontName='Helvetica', fontSize=10,
                                   leading=12, wordWrap='CJK', autoLeading='max')
            paragraph = Paragraph('中文段落' * 5 + f'<img src="{image}" width="55" height="25"/>' + '中文后文' * 5, style)
            width, height = paragraph.wrap(90, 700)
            self.assertGreater(height, 25)
            # Each image is an atomic box, not an empty character passed to ord().
            self.assertTrue(all(line.extraSpace >= -10 for line in paragraph.blPara.lines))

    def test_non_math_document_needs_no_node(self):
        with tempfile.TemporaryDirectory() as folder:
            md = Path(folder) / 'report.md'
            md.write_text('# 普通日报\n\n```python\nprice = "$x$"\n```\n')
            self.assertEqual(math.MathTypesetter(md).items, {})

    def test_math_markup_preserves_prose_and_baseline(self):
        typesetter = object.__new__(math.MathTypesetter)
        typesetter.folder = Path('/tmp/formula-assets')
        typesetter.items = {('x_t', False): dict(png='x.png', width=20, height=16, depth=3)}
        result = typesetter.inline('概率$x_t$后文', lambda x: x)
        self.assertIn('valign="-1.875"', result)
        self.assertTrue(result.startswith('概率<img'))
        self.assertTrue(result.endswith('/>后文'))
