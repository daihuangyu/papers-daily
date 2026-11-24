# PDF 与数学排版交付检查

最终 PDF 21 页，来自同一份完整 report.md（28741 字符）。

- 314 段非数学可见正文、代码及表格片段逐段对照 PDF 提取文本；无缺漏。独立数学块不以文本提取代替可视验证。
- 19 张原图表逐张核对 PDF 解码像素与 assets PNG 一致；源页、版本和裁剪坐标见 SOURCES.md。初轮裁剪不全处已修正，最终版保留图例、坐标轴及表头末行。
- 115 个不同公式、120 处渲染；19 个独立公式为 SVG 矢量路径，96 个不同行内公式以 576-DPI 透明 PNG 按基线排版。TeX 全部通过 MathJax，源码哈希一致，无解析后未使用公式。
- 所有 21 页由 Poppler 实际渲染为 PNG，逐页检查四张接触表，并检查完整分辨率公式样例；未见重叠、越界、缺图或空白页。
- pdfplumber 检查文字与图片边界通过；TrueType 子集字体嵌入并带 ToUnicode，全文字符预检通过。
- 本地适配入口 assets/render_report.py 调用现有共享 renderer；可设 PAPERS_DAILY_HOST_FONT 使用可嵌入 CJK TrueType。未修改共享脚本。Node 依赖由仓库 package-lock 固定。
- 未运行论文模型训练；教学算例不作为实验结果。GitHub 实际数学显示与远端文件完整性在推送后单独核验。

Markdown SHA-256：`749a35777e3f12a11ba4dbccb4f00c6258b68356af3d6781c58d943aae4b29e8`

PDF SHA-256：`aefe1463d2e733e85ad001fddfff21f18ea36467a7bcffca57993455814610ce`

机器核查记录见 pdf-check.json；页面PNG保留本地研究工作目录。
