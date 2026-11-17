"""Render the complete Markdown, including original figures, to PDF."""
import argparse
import html
import re
from pathlib import Path
from cjk_font import embedded_font
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]

def illustrations(folder):
    FONT = embedded_font((folder / 'report.md').read_text(encoding='utf-8') + '教学示意：机架共享状态与独立执行池共享缓存输入输出复用标准验证接受前缀拒绝后残差采样更新上下文进入下一轮目标保持分布非原文测量图')
    assets = folder / 'assets'
    assets.mkdir(exist_ok=True)
    def chart(name, title, boxes, note):
        image = PILImage.new('RGB', (1600, 570), '#f3f7fb')
        d = ImageDraw.Draw(image)
        f = ImageFont.truetype(str(FONT), 32, index=0)
        ft = ImageFont.truetype(str(FONT), 42, index=0)
        d.text((55,35), title, font=ft, fill='#123148')
        for i, text in enumerate(boxes):
            x = 55 + i*390
            d.rounded_rectangle((x,165,x+320,375),radius=20,fill='#e1eff1',outline='#247080',width=3)
            d.multiline_text((x+22,210),text,font=f,fill='#123148',spacing=15)
            if i < len(boxes)-1:
                d.line((x+325,270,x+378,270),fill='#247080',width=5)
                d.polygon([(x+378,270),(x+360,260),(x+360,280)],fill='#247080')
        d.text((55,455),note,font=f,fill='#516477')
        image.save(assets/name)
    chart('rubin-teaching.png','教学示意：机架共享状态与独立执行池',[
        'Dynamo 路由\n估计未缓存工作', 'Prefill 池\n查找 / 加载 KV', 'Decode 池\nTP 或 Wide-EP', '输出 tokens\n复用共享状态'], '共享 Mooncake KV pool；Engram 表在 HBM 分片。非原文测量图。')
    chart('speculative-teaching.png','教学示意：标准 speculative sampling',[
        'Draft 采样\nx ~ q(x)', 'Target 验证\nmin(1, p/q)', '连续接受前缀\n拒绝后残差采样', '更新上下文\n进入下一轮'], '目标：保持 target 分布；可选有损验证不具备同样的等价保证。')

def build(md):
    FONT = embedded_font(md.read_text(encoding='utf-8'))
    pdfmetrics.registerFont(TTFont('CJK',str(FONT)))
    pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK',italic='CJK',boldItalic='CJK')
    normal = ParagraphStyle('body',fontName='CJK',fontSize=10,leading=17,spaceAfter=8,wordWrap='CJK')
    headings = {n: ParagraphStyle(f'h{n}',parent=normal,fontSize={1:24,2:17,3:13}[n],leading={1:32,2:25,3:20}[n],textColor=colors.HexColor('#164e63'),spaceBefore=12,spaceAfter=9,keepWithNext=True) for n in (1,2,3)}
    code = ParagraphStyle('code',parent=normal,fontName='CJK',fontSize=8,leading=12,backColor=colors.HexColor('#eef3f7'),borderPadding=7)
    caption = ParagraphStyle('caption',parent=normal,fontSize=8,leading=12,textColor=colors.HexColor('#536575'))
    def inline(text):
        text=html.escape(text)
        text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#126b86">\1</link>',text)
        text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
        text=re.sub(r'`([^`]+)`',r'\1',text)
        return text
    lines=md.read_text(encoding='utf-8').splitlines(); story=[]; i=0
    while i<len(lines):
        line=lines[i].strip(); i+=1
        if not line: continue
        if line.startswith('```'):
            block=[]
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(html.escape(lines[i]).replace(' ','&#160;')); i+=1
            i+=1
            story.append(Paragraph('<br/>'.join(block),code)); story.append(Spacer(1,8)); continue
        image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if image:
            path=md.parent/image[2]
            w,h=PILImage.open(path).size
            scale=min(487/w,540/h)
            story.append(Image(str(path),width=w*scale,height=h*scale)); story.append(Spacer(1,7)); continue
        heading=re.match(r'^(#{1,3}) (.+)',line)
        if heading:
            n=len(heading[1]); title=heading[2]
            story.append(Paragraph(inline(title),headings[n])); continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip()); i+=1
            cells=[]
            for row in rows:
                if re.fullmatch(r'[| :\-]+',row): continue
                cells.append([Paragraph(inline(x.strip()),caption) for x in row.strip('|').split('|')])
            table=Table(cells,colWidths=[487/len(cells[0])]*len(cells[0]),repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1eff1')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#cad7df')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,10)]); continue
        if line=='---': continue
        story.append(Paragraph(inline(line),caption if line.startswith('图 ') else normal))
    def footer(c,d):
        c.setFont('CJK',8); c.setFillColor(colors.HexColor('#536575'))
        c.drawString(54,26,'Papers Daily / VLM / '+md.parent.name)
        c.drawRightString(541,26,str(d.page))
    output=md.with_suffix('.pdf')
    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=48,title=lines[0].lstrip('# '),author='Papers Daily').build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('markdown',type=Path); p.add_argument('--illustrations',action='store_true'); a=p.parse_args()
    if a.illustrations: illustrations(a.markdown.parent)
    build(a.markdown)

