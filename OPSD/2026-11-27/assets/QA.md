# 2026-11-27 渲染与内容检查

- PDF 从本日完整 Markdown 生成，中文字体已嵌入并含 Unicode 映射。PDFium 渲染全部 15 页，检查中文、公式、代码、图表和页边界。
- 按 Markdown 比对 284 段文字、表格单元格和伪代码片段。公式清单与正文全部 172 处数学一致，其中 14 处独立公式。公式渲染见 math-manifest.json。
- 17 个原文图表裁图逐一核验，嵌入 PDF 中的像素与提交文件相符。图号、页码和裁剪信息见 SOURCES.md 和 manifest.json。
- 五篇速读均在 PDF 第一页，顺序跳往第 1, 5, 6, 10, 12 页的对应精读。链接页内目标和标题均已核对。
- Markdown SHA-256 为 18dfe666a17baf90c2bf08167accdfff39639cffcd7dde3d772f59c224603d6f。
- PDF SHA-256 为 0789900193b698ef9dd4b53306bbd57351d575ab4e8decdb40890de050bdf4d2。
