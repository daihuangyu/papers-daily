"""Prepare an embedded TrueType CJK font from Linux Noto (no system credentials)."""
import hashlib
import os
import tempfile
from pathlib import Path

from fontTools import subset
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont


def embedded_font(text):
    """Subset Noto's CFF outlines and convert to TrueType for ReportLab embedding."""
    override = os.environ.get('PAPERS_CJK_FONT')
    source = Path(override or '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not source.is_file():
        raise FileNotFoundError('Install fonts-noto-cjk or set PAPERS_CJK_FONT to an embeddable Chinese font')
    index = int(os.environ.get('PAPERS_CJK_FONT_INDEX', '2' if source.suffix == '.ttc' and not override else '0'))
    # Include footer, replacement characters and ASCII used by the renderer.
    chars = ''.join(sorted(set(text + 'Papers Daily / VLM / 0123456789-中文图公式' + ''.join(map(chr, range(32, 127))))))
    digest = hashlib.sha256((str(source) + str(source.stat().st_mtime_ns) + str(index) + chars).encode()).hexdigest()[:24]
    cache = Path(tempfile.gettempdir()) / 'papers-daily-fonts'
    cache.mkdir(exist_ok=True)
    output = cache / (digest + '.ttf')
    if output.exists():
        return output
    font = TTFont(source, fontNumber=index)
    missing = sorted({ord(c) for c in chars if not c.isspace()} - set(font.getBestCmap()))
    fallback = TTFont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf') if missing else None
    unsupported = [c for c in missing if c not in fallback.getBestCmap()]
    if unsupported:
        raise ValueError('Fonts are missing characters: ' + ' '.join(f'U+{c:04X}' for c in unsupported[:30]))
    options = subset.Options()
    options.layout_features = []
    sub = subset.Subsetter(options=options)
    sub.populate(text=chars)
    sub.subset(font)
    if 'glyf' in font and not missing:
        font.save(output)
        font.close()
        return output
    order = font.getGlyphOrder()
    glyphset = font.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyphset)
        glyphset[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    cmap = font.getBestCmap()
    metrics = {g: font['hmtx'].metrics[g] for g in order}
    if fallback:
        scale = font['head'].unitsPerEm / fallback['head'].unitsPerEm
        fs = fallback.getGlyphSet()
        for char in missing:
            original = fallback.getBestCmap()[char]
            name = f'fallback_{char:X}'
            recording = DecomposingRecordingPen(fs)
            fs[original].draw(recording)
            pen = TTGlyphPen(None)
            recording.replay(TransformPen(pen, (scale, 0, 0, scale, 0, 0)))
            glyphs[name] = pen.glyph()
            metrics[name] = tuple(round(v * scale) for v in fallback['hmtx'].metrics[original])
            order.append(name)
            cmap[char] = name
        fallback.close()
    builder = FontBuilder(font['head'].unitsPerEm, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap(cmap)
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=font['hhea'].ascent, descent=font['hhea'].descent)
    builder.setupNameTable({'familyName': 'PapersDaily CJK', 'styleName': 'Regular',
        'uniqueFontIdentifier': 'PapersDaily CJK ' + digest, 'fullName': 'PapersDaily CJK Regular',
        'psName': 'PapersDailyCJK-Regular', 'version': 'Version 1.0'})
    os2 = font['OS/2']
    builder.setupOS2(sTypoAscender=os2.sTypoAscender, sTypoDescender=os2.sTypoDescender,
        usWinAscent=os2.usWinAscent, usWinDescent=os2.usWinDescent)
    builder.setupPost()
    builder.setupMaxp()
    builder.save(output)
    font.close()
    return output
