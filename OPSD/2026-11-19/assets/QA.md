# 2026-11-19 全文 PDF 与发布前检查

检查日期：2026-11-19，Asia/Shanghai。本记录说明实际完成的检查，不代表已运行各论文模型训练。

- 选题：5 篇一手论文，首次公开日期均在 2026-10-19 至 2026-11-19；与 sources.json 中 40 条来源及全部 8 份历史 manifest 按去版本 arXiv、DOI、规范 URL、标题与别名交叉检查，无重复。含当天已提交的 VLM 内容；ECOD 的一手 v1 日期为 10-17，已排除。
- 原文：使用指定 v1/v2 PDF，逐篇读取相关方法、关键公式、主结果及所需附录配方/证明；PDF 文件总计 178 页，不把这一页数表述为逐页完整复现全部附录。正文区分原文、静态代码核查、教学推演和未披露项。
- 原图：19 幅来自原 PDF 的架构/机制图与实验图表。已检查源页、裁切预览及最终 PDF；裁切保留完整图形、坐标、图例、表格与对应原图注，无重绘或伪造实验。页码、版本、裁切坐标和 SHA-256 见 SOURCES.md 与 manifest。
- 代码：仅 SWE-OPD 找到并检查了专门作者实现，固定提交 0884a04f4c9e2b4baaa47261b2016d425ac3f63e。论文 detached normalizer 与代码可微分母、简单 token_reverse_kl selector 与 PG selector 的差别已在正文披露。未运行 GPU 训练；其余四篇没有核验到专门作者方法代码，不把框架或第三方适配器冒称作者实现。
- 内容一致：report.pdf 从最终 report.md 全文生成；Markdown 32,994 字符，逐项比对 280 条非空正文、公式、代码及表格行，去除 Markdown 标记和链接目标并归一化空白后全部在 PDF 文本中找到，缺失 0 条。
- 图片一致：PDF 中 19 个原图表对象，与 assets 中 19 幅 PNG 解码后的 RGB 像素哈希逐个匹配，未发生缩图换内容或漏图。
- 字体：在当前 macOS 本地环境使用已有可嵌入的 Songti TrueType 字体；4 个 STSongti-TC-Bold 字体子集均有 FontFile2 和 ToUnicode。最终 Markdown 全部可见字符通过 cmap 检查，缺字 0。梯度和转置使用正文明确解释的 grad、^T，避免缺字。此运行未声称使用 Linux Noto。
- 排版：25 页 A4。全 25 页以 1.5× 渲染，查看覆盖全部页的 contact sheets；另放大检查公式与密集表格页 3、9、10、20、21。文字边界检查无越界，无空白页、丢字、图注截断或可见重叠；已收紧原表裁切，放大 Agentic Table 1，并消除末尾仅有一段的多余页面。
- 生成：使用仓库 scripts/build_report.py 的同一 Markdown 解析器，仅通过本日报 assets/render_report.py 调整本地可用字体、行距/段距与图片高度。共享脚本及其他方向内容未改动。可选设置 PAPERS_DAILY_HOST_FONT 后执行该脚本；不设置则走仓库原有字体准备流程。当前运行字体 SHA-256、renderer/adapter SHA-256 见 pdf-check.json。
- 基础链接：GKD、SWE-bench、ALFWorld、tau2-bench、VERL 官方资料和作者主页均仅为辅助，不占 5 篇。没有找到独立作者博客的条目已直说。当前 VERL 文档版本与论文所用 v0.8 的差异有注明。
- 仓库检查：现有 scripts/tests 的 10 项测试通过；索引由 scripts/update_index.py 重建。仅提交本日报目录、OPSD/README.md、根 README.md 与 sources.json。SSH 发布、禁止 force push 的授权来自本次任务指令；AUTOMATION.md 的 VLM 云端流程没有被修改。

机器检查的页数、逐页文字量、全部可见行比对、图片匹配、字体嵌入与文件哈希见 [pdf-check.json](pdf-check.json)。远端提交与文件完整性在 push 后另行核验，并在任务回执提供 commit 与链接。
