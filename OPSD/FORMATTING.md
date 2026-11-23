# OPD / OPSD 数学排版

正文采用 GitHub 原生 LaTeX 数学语法；Markdown 与 PDF 共用一份数学源码。

- 短式使用 `$p_\theta(v\mid s)$`，不要以 `p_theta`、`^T` 或 Unicode 拼接替代正式公式。
- 独立公式使用单独一行 `$$` 开始和结束，中间写 LaTeX；长式用 `aligned` 分行。数学块两侧留空行。
- 分式使用 `\frac`，求和与期望写清下标；矩阵转置用 `^\top`。逐符号解释放在公式附近。
- 公式只转换排版，不改写论文结论。原文有矛盾时保留原式并明确标注；本文推演与候选修正分别说明。
- 原论文架构和实验图保留真实截图。`assets/math/` 是正文公式排版缓存，不是原论文实验图；两类素材分别计数。

首次准备可复现的可选 Node 依赖：

```sh
npm ci --prefix scripts/mathjax
python3 scripts/build_report.py OPSD/YYYY-MM-DD/report.md
```

既有日报的本机字体适配入口 `assets/render_report.py` 仍可使用。自定义独立依赖目录可设 `PAPERS_MATH_NODE_MODULES`；默认使用 `scripts/mathjax/node_modules`，不依赖网络公式服务。无数学标记的旧文档不会启动 MathJax。

PDF 用 MathJax 输出 SVG，再以四倍像素密度嵌入；行内公式按基线对齐，行距自动容纳分式。GitHub 页面直接渲染同一 LaTeX；SVG/PNG 与公式源码映射留在当日报告目录，便于离线查看和核查。

每次生成后重新检查全文、全部公式和原图，实际渲染每一页；不得仅核对文字提取。更新 manifest 的 Markdown/PDF/公式素材哈希与 QA 记录。
