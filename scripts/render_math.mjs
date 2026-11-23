// Render the same $...$ / $$...$$ source used by GitHub Markdown.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
const require = createRequire(process.env.PAPERS_MATH_NODE_MODULES
  ? path.resolve(process.env.PAPERS_MATH_NODE_MODULES, '../package.json')
  : new URL('./mathjax/package.json', import.meta.url));
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const {Resvg} = require('@resvg/resvg-js');
const adaptor=liteAdaptor(); RegisterHTMLHandler(adaptor);
const input=new TeX({packages:AllPackages.filter(p=>!['noerrors','noundefined'].includes(p)),
  formatError:(_jax,error)=>{throw error;}});
const doc=mathjax.document('', {InputJax:input, OutputJax:new SVG({fontCache:'none'})});
const [requestFile, outputDir] = process.argv.slice(2);
fs.mkdirSync(outputDir,{recursive:true});
const results=[];
for (const item of JSON.parse(fs.readFileSync(requestFile,'utf8'))) {
  const key=crypto.createHash('sha256').update(JSON.stringify([item.tex,item.display])).digest('hex').slice(0,20);
  const node=doc.convert(item.tex,{display:item.display,em:16,ex:8,containerWidth:780});
  if (adaptor.outerHTML(node).includes('data-mml-node="merror"')) throw Error(`Invalid TeX: ${item.tex}`);
  const svgNode=adaptor.firstChild(node);
  let svg=adaptor.outerHTML(svgNode);
  const viewBox=adaptor.getAttribute(svgNode,'viewBox').split(/\s+/).map(Number);
  const width=viewBox[2]*16/1000, height=viewBox[3]*16/1000;
  const style=adaptor.getAttribute(svgNode,'style') || '';
  const match=style.match(/vertical-align:\s*([-\d.]+)ex/);
  const depth=match ? -Number(match[1])*8 : 0;
  svg=svg.replace(/width="[^"]+"/,`width="${width*4}"`).replace(/height="[^"]+"/,`height="${height*4}"`);
  fs.writeFileSync(path.join(outputDir,key+'.svg'),svg);
  if (!fs.existsSync(path.join(outputDir,key+'.png')))
    fs.writeFileSync(path.join(outputDir,key+'.png'),new Resvg(svg,{font:{loadSystemFonts:false}}).render().asPng());
  results.push({...item,key,width,height,depth,png:key+'.png',svg:key+'.svg'});
}
fs.writeFileSync(path.join(outputDir,'index.json'),JSON.stringify(results,null,2)+'\n');
