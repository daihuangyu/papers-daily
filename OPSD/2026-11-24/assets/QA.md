# PDF 与数学排版交付检查

最终 PDF 22 页，来自同一份完整 report.md（30100 字符）。首页放入五篇速读，五个链接分别跳到对应精读的标题。

- 331 段非数学可见正文、代码及表格片段逐段对照 PDF 提取文本；无缺漏。独立数学块不以文本提取代替可视验证。
- 19 张原图表逐张核对 PDF 解码像素与 assets PNG 一致；源页、版本和裁剪坐标见 SOURCES.md。初轮裁剪不全处已修正，最终版保留图例、坐标轴及表头末行。
- 115 个不同公式、120 处渲染；19 个独立公式为 SVG 矢量路径，96 个不同行内公式以 576-DPI 透明 PNG 按基线排版。TeX 全部通过 MathJax，源码哈希一致，无解析后未使用公式。
- 所有 22 页由 Poppler 实际渲染为 PNG，逐页检查四张接触表，并以完整分辨率检查新增速读首页；未见重叠、越界、缺图或空白页。
- pdfplumber 检查文字与图片边界通过；TrueType 子集字体嵌入并带 ToUnicode，全文字符预检通过。
- 本地适配入口 assets/render_report.py 调用现有共享 renderer；可设 PAPERS_DAILY_HOST_FONT 使用可嵌入 CJK TrueType。共享脚本支持 Markdown 标题前的显式锚点与对应 PDF 跳转。Node 依赖由仓库 package-lock 固定。
- 未运行论文模型训练；教学算例不作为实验结果。GitHub 实际数学显示与远端文件完整性在推送后单独核验。

Markdown SHA-256：`9267a58a3aa4ee9b50b24c3eed02bc03f0c2910b6943eaf3d9cd588e5ccb2e69`

PDF SHA-256：`d22474963380ecad7a38685a8f4d43152cb3b259c775e9f6fdfa967e3110933c`

机器核查记录见 pdf-check.json；页面PNG保留本地研究工作目录。
