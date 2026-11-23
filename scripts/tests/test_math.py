import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from math_render import MathRenderer, MathParagraph, inline_math, math_blocks
from reportlab.graphics import renderPDF
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas


class MathTests(unittest.TestCase):
    def test_literal_code_and_escaped_currency_are_not_math(self):
        self.assertEqual(list(inline_math(r'cost \$20, `$not_math$`, $x_i$')),['x_i'])
        self.assertEqual(list(inline_math(r'$`a_b`$ and `$literal$` then $`\mathrm{mean}(R)`$')),['a_b',r'\mathrm{mean}(R)'])
        lines=['```python','$$not math$$','```','$$','a+b','$$','```math','c+d','```']
        self.assertEqual(list(math_blocks(lines)),[(3,6,'a+b'),(6,9,'c+d')])

    def test_unclosed_display_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unclosed'):
            list(math_blocks(['$$','a+b']))

    def test_github_rejected_macro_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            md=Path(temp)/'report.md';md.write_text(r'$`\operatorname{mean}(R)`$')
            with self.assertRaisesRegex(ValueError,'GitHub rejects'):
                MathRenderer(md)

    def test_html_sensitive_comparison_must_be_escaped(self):
        with tempfile.TemporaryDirectory() as temp:
            md=Path(temp)/'report.md';md.write_text(r'$`y_{<t}`$')
            with self.assertRaisesRegex(ValueError,'literal <'):
                MathRenderer(md)

    def test_vector_display_inline_baseline_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            md=Path(temp)/'report.md'
            md.write_text('Inline $`x_i`$.\n\n```math\n\\frac{1}{2}+\\sum_{i=1}^n x_i\n```\n')
            math=MathRenderer(md)
            tex=math.blocks[2][1]
            drawing=math.display(tex)
            self.assertTrue(drawing.contents)
            self.assertLess(drawing.width,487)
            stream=io.BytesIO();canvas=Canvas(stream)
            renderPDF.draw(drawing,canvas,20,600)
            # Force an inline image onto a CJK line boundary (ReportLab regression).
            content=math.inline('long-prefix $`x_i`$ trailing text',lambda x:x)
            para=MathParagraph(content,ParagraphStyle('test',wordWrap='CJK',fontSize=10,leading=17,autoLeading='max'))
            width,height=para.wrap(65,500)
            self.assertGreater(height,17)
            para.drawOn(canvas,20,500);canvas.save()
            self.assertTrue(stream.getvalue().startswith(b'%PDF'))
            math.write_manifest()
            data=json.loads((md.parent/'assets/math-manifest.json').read_text())
            self.assertEqual(data['rendered_occurrences'],2)
            self.assertEqual(len(data['expressions']),2)

    def test_invalid_tex_does_not_silently_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            md=Path(temp)/'report.md'
            md.write_text('$$\n\\frac{1}{\n$$\n')
            with self.assertRaisesRegex(RuntimeError,'Math rendering failed'):
                MathRenderer(md)

    def test_legacy_report_does_not_need_math_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            md=Path(temp)/'report.md';md.write_text('# Legacy\n```text\nx = y\n```\n')
            self.assertFalse(MathRenderer(md).entries)


if __name__=='__main__':unittest.main()
