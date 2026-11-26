# 2026-11-26 渲染与内容检查

- PDF 从本日完整 Markdown 生成，中文字体已嵌入并含 Unicode 映射。PDFium 渲染全部 15 页，检查中文、公式、代码、图表和页边界。
- 按 Markdown 比对 284 段文字、表格单元格和伪代码片段。公式清单与正文全部 157 处数学一致，其中 13 处独立公式。公式渲染见 math-manifest.json。
- 17 个原文图表裁图逐一核验，嵌入 PDF 中的像素与提交文件相符。图号、页码和裁剪信息见 SOURCES.md 和 manifest.json。
- 五篇速读均在 PDF 第一页，顺序跳往第 2, 4, 7, 9, 12 页的对应精读。链接页内目标和标题均已核对。
- Markdown SHA-256 为 26eb00fd4e8fe9bffcd8f642f033efecf8e3e00e055130c33321530af100d950。
- PDF SHA-256 为 dcb8346407569daec0bd3125d971242d9184b359556d57d75e93f7609a8c5825。
