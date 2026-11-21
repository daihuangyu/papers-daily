# PDF 与图文交付检查

检查日期：2026-11-21。最终 PDF 22 页，来自同一份完整 report.md。

- 32630 个 Markdown 字符，198 条可见非空行逐行对照 PDF 提取文字；未漏行，公式、代码、表格均在全文中。
- 19 张真实论文原图表全部嵌入；逐图解码像素与 assets PNG 相同，不是链接占位符。源PDF裁图坐标与哈希见 SOURCES.md/manifest。
- 22 页全部实际渲染为 PNG 并检查接触表；最终重排后的13—22页重新看过。未发现缺字、越界、遮挡、图注截断或空白页；版面分页为正文与完整图组保留空间。
- 字体 TrueType 子集已嵌入，带 ToUnicode 映射；渲染前检查全文字符都在字体 cmap 内。数学采用可搜索 Unicode/线性表达，grad、^T 等记法均有解释。
- pdfplumber 检查字符边界未越过正文和页脚允许区域。原图裁切曾发现半截图注/表格末行，已修正后重建最终PDF。
- 本机字体适配通过 assets/render_report.py；可用 PAPERS_DAILY_HOST_FONT 指向可嵌入CJK TrueType，或不设置时采用仓库原有Linux字体准备流程。未改动共享renderer。
- 未运行任何论文大模型训练；本文推导/算例不是复现实验结果。代码公开状态、配置差异、原文未披露的信息在正文分别标注。

Markdown SHA-256：`e65a28c018a93ccdb0d8aee4e4f6a57de3e272e597a2b384e7b771a38a8c553e`

PDF SHA-256：`a7d9cf119c7ec88903fa64f5df109aa02f0e2988fe2bf34d4784401cbcd36e84`

详尽机器检查记录见 [pdf-check.json](pdf-check.json)。页面渲染PNG保留在本次研究工作目录，不作为新实验图上传。
