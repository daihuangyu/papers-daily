# 数学公式排版

Markdown 是唯一内容源。正文优先用 GitHub 受保护的行内写法 ``$`x_i`$``，独立公式优先 fenced `math`，避免 Markdown 先处理下划线、反斜杠和行间距指令。PDF 也支持普通 `$...$` 和 `$$`，但发布前必须检查 GitHub 实际显示。多行用 `aligned`；代码示例仍放普通代码块，不把推导写成伪代码截图。函数名使用 `\mathrm{mean}`、`\mathrm{clip}` 等；GitHub 实测拒绝 `\operatorname`，生成器会报错提示替换。小于号使用 `\lt`（如 `y_{\lt t}`），避免网页将字面 `<t` 当作 HTML 后破坏公式。

````markdown
行内说明：$`\pi_\theta`$ 是策略，$`\pi_{\mathrm{ref}}`$ 是参考模型。

```math
\begin{aligned}
z&=\beta(u_w-u_l),\\
\mathcal{L}&=-\mathbb{E}[\log\sigma(z)].
\end{aligned}
```
````

首次安装：`python3 -m pip install -r requirements.txt`、`npm ci --ignore-scripts`（Node.js 20+）。随后执行 `python3 scripts/build_report.py VLM/YYYY-MM-DD/report.md`。保留原有中文字体环境设置，Linux 仍用 Noto，数学字形独立由 MathJax 提供。依赖安装完成后，排版不访问 CDN、浏览器或网络。

GitHub 直接渲染 Markdown 中的 LaTeX。PDF 使用同样的 TeX：独立公式转换为 SVG 矢量路径，行内公式使用透明 576-DPI PNG 并按数学基线对齐；中文自动行距防止分数和上下标相撞。公式无法解析或太宽会报错，不能退回原始 TeX 后照常发布。长公式应拆为多行，不缩小到难以阅读。

生成器在当日 `assets/math-manifest.json` 记录公式源、渲染次数、尺寸、SVG/行内 PNG 哈希与 Markdown 哈希；临时图形缓存在系统临时目录，不将缓存或 node_modules 提交。PDF 中的公式图形不能用普通文本提取完整还原，全文检查应将数学片段与非数学正文分别核验；逐式检查 TeX/渲染清单、逐段检查正文、逐页检查排版，并保留原有论文图。

运行 `python3 -m unittest discover -s scripts/tests -v`，包含中文行内图换行、错误公式阻止发布、代码字面量保护和渲染清单校验。旧报告的普通文字不会自动猜测改为 LaTeX；历史稿件需明确转换和重新检查后再发布。

规范依据：[GitHub 数学表达式官方文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)、[MathJax 3.2 文档](https://docs.mathjax.org/en/v3.2/)。为固定排版，本仓库锁定 MathJax 3.2.2 和 sharp 版本，升级时重新检查公式样例及分页。
