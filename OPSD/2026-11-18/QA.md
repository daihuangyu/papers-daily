# 2026-11-18 OPD & OPSD 日报检查记录

- 近期窗口 2026-10-18 至 2026-11-18；5 篇不同 primary papers，全部阅读 v1，并核对首次公开历史。博客、教程、代码不占名额。
- 去重读取 sources.json 和全部 5 份历史 manifest，共 25 条历史来源，包含当日已发布的 VLM/2026-11-18。对照去版本 arXiv、DOI、规范 URL、规范标题与近似标题人工复核，无重复。索引最终为 6 份日报、30 条唯一来源、145 个别名。
- report.md 为 PDF 唯一正文来源：27355 个 Unicode 字符，19 页，16 个原始图片对象；208 个非空正文、表格和代码行逐行规范化空白后均在 PDF 文字层找到，缺行数 0。公式以可读 Unicode/线性数学记号完整排版，没有未渲染的 LaTeX。
- 全部 19 页由 PDFium 以 1.5 倍渲染，逐页检查联系表，并对第 4/10/14/16 页全尺寸检查图表、算法、曲线、公式、中文和代码。没有空白页、裁断的主图、正文重叠或缺字；图和对应中文图注成组。16 张原图裁切均检查，裁边坐标和 SHA-256 存于 manifest。
- 字体 cmap 对正文全字符检查：缺字 0。正文使用本机已有、转为可嵌入 TrueType 的 Songti 字体；四个 TrueType 子集均嵌入并带 Unicode 映射。PDF 资源另含未用于正文的 Helvetica 默认资源，不将其误报为已嵌入中文字体。逐字符边界检查出页数 0。
- 使用仓库共享 renderer 的本地适配：OPSD 页脚、已有嵌入字体、图片高度上限 300 pt 和图注同页；没有修改共享 Linux renderer、AUTOMATION.md 或 VLM 内容。没有宣称云端迁移完成。
- MTA 官方仓库固定提交 9567be28a9882f6fa3c3f610f358beb07bb29be8；读取 README、配置、启动脚本、actor 和实际 core_algos 调用链。论文 RKL 与默认 beta=0 的 FKL 差异已明确记录，原运行覆盖参数/日志未知。其余四篇本次未找到专用作者实现；未用基础框架冒充作者代码。
- TESD 主文与附录奖励加减号及策略目标符号不一致均保留；各论文 train/test thinking、评估 Avg/Pass、checkpoint 选择和非等预算基线分开解释。数学算例使用 Python 重算。没有训练模型，没有生成或重绘实验图。
- Markdown SHA-256：e309b7499a557fe3244d585d3a57f09ab9653bd619e37e00a0e436740eff2df9
- PDF SHA-256：fa102da99856eba98df3c392166cd562e3fa0a0c071f2f7e9b8bcb2060c64eba

机器检查记录见 assets/pdf-check.json，全部原始图表来源见 assets/SOURCES.md。
