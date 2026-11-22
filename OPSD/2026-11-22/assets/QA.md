# PDF 与图文交付检查

检查日期：2026-11-22。最终 PDF 22 页，来自同一份完整 report.md。

- 29316 个 Markdown 字符，185 条可见非空行逐行对照 PDF 提取文字；未漏行，公式、代码、表格均在全文中。
- 20 张真实论文原图表全部嵌入；逐图解码像素与 assets PNG 相同，不是链接占位符。源PDF裁图坐标与哈希见 SOURCES.md/manifest。
- 22 页全部实际渲染为 PNG 并检查接触表；另以完整分辨率检查第19页公式与推导样例。未发现缺字、越界、遮挡、图注截断或空白页；版面分页为正文与完整图组保留空间。
- 字体 TrueType 子集已嵌入，带 ToUnicode 映射；渲染前检查全文字符都在字体 cmap 内。数学采用可搜索 Unicode/线性表达，grad、^T 等记法均有解释。
- pdfplumber 检查字符边界未越过正文和页脚允许区域。原图裁切曾发现半截图注/表格末行，已修正后重建最终PDF。
- 本机字体适配通过 assets/render_report.py；可用 PAPERS_DAILY_HOST_FONT 指向可嵌入CJK TrueType，或不设置时采用仓库原有Linux字体准备流程。未改动共享renderer。
- 未运行任何论文大模型训练；本文推导/算例不是复现实验结果。代码公开状态、配置差异、原文未披露的信息在正文分别标注。

Markdown SHA-256：`de4e28c7f096ef1b045b8be02d9a17269c0e006613a514c021620b84a42b2311`

PDF SHA-256：`453b0eb000214050b92d31ea2d161d6aea73bf9b9cb09bc5cb6462cf4896757b`

详尽机器检查记录见 [pdf-check.json](pdf-check.json)。页面渲染PNG保留在本次研究工作目录，不作为新实验图上传。
