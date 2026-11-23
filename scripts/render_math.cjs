// Local, deterministic TeX rendering; no browser, CDN or network at build time.
const fs = require('node:fs');
const path = require('node:path');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const sharp = require('sharp');

async function main() {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  fs.mkdirSync(input.directory, {recursive: true});
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const tex = new TeX({packages: AllPackages,
    formatError: (_jax, error) => {throw error;}});
  const document = mathjax.document('', {
    InputJax: tex, OutputJax: new SVG({fontCache: 'none'})});
  const result = [];
  for (const item of input.expressions) {
    const node = document.convert(item.tex, {display: item.display});
    const svgNode = adaptor.firstChild(node);
    let svg = adaptor.outerHTML(svgNode);
    if (svg.includes('data-mml-node="merror"'))
      throw new Error('Invalid TeX: ' + item.tex);
    const box = adaptor.getAttribute(svgNode, 'viewBox').split(/\s+/).map(Number);
    const width = box[2] * item.size / 1000;
    const height = box[3] * item.size / 1000;
    const depth = Math.max(0, box[1] + box[3]) * item.size / 1000;
    if (![width,height,depth].every(Number.isFinite) || width <= 0 || height <= 0)
      throw new Error('Invalid math dimensions: ' + item.tex);
    svg = svg.replace(/width="[^"]*"/, `width="${width}pt"`)
      .replace(/height="[^"]*"/, `height="${height}pt"`)
      .replace(/currentColor/g, '#163342');
    const base = path.join(input.directory, item.id);
    fs.writeFileSync(base + '.svg', svg);
    // Inline formulae use transparent 576-DPI assets (8 pixels per PDF point).
    // Display formulae are embedded as vector paths from the SVG, not this PNG.
    await sharp(Buffer.from(svg), {density: 576}).png().toFile(base + '.png');
    result.push({...item, width, height, depth});
  }
  process.stdout.write(JSON.stringify(result));
}
main().catch(error => {console.error(error.message); process.exit(1);});
