# 2026-11-17 OPSD 日报检查记录

- 时间窗2026-10-17至2026-11-17，5篇primary papers；逐篇核对arXiv首次提交历史和阅读版本。C-OPSD阅读v2，但其首次提交也在窗口内，此前未收录。
- 读取全仓sources.json及全部历史manifest，包含VLM/2026-11-17；update_index.py验证4份日报、20个唯一来源、90个去重别名。
- 23393字符Markdown为PDF唯一正文来源。最终PDF20页、12个原始图表嵌入对象；169个非空正文/表格/代码行逐行核对，规范化空白后均在PDF中找到，无缺行。
- 使用字体cmap逐字核查中文、希腊字母及数学记号均有字形；PDF文字边界检查无出页字符。未把未渲染LaTeX当公式排版，公式使用可读纯文本数学记号和对应原文图表。
- 最终20页逐页渲染，并检查全部页面缩略图及12幅原图裁切：无空白页、表格/图形裁断、正文重叠；原图来源与页码见SOURCES.md。未重绘实验图。
- 本次在已有本地日程中生成；中文使用已转为可嵌入TrueType的macOS Songti字体。保留仓库现有Linux云端renderer和VLM资料，未宣称云端迁移已完成。
- 论文代码核验：Shuffled原PDF链接Pear3on/S_OPSD本次返回HTTP404，正文承诺接收后公开；其余四篇没有定位到可核验的作者专属实现。通用基础仓库不冒充作者代码。
- 数字边界已保留：ECOS主表引用基线与本地消融聚合不同；TOPSD反向KL和验证开销；Shuffled单主训练seed与带符号cap；Async总实现包含batching/export变化；C-OPSD各benchmark选择最好checkpoint及GB300特定成本。
- 没有训练模型。数值算例、梯度推导、GPU-hours换算与方法组合建议均明确标为本文推导。
- PDF SHA256：d6c47c2b8290e4116ce414f676112df252b5248ffe62b478056127d6092db496。
