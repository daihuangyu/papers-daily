# PDF 全文与公式检查

报告日期 2026-11-22。2026-11-24 发布前重新排版并检查全部 28 页。

- 完整阅读五篇正式主会论文及全部附录，共125页：MemAgent29、DECS30、Speculative Actions24、R-Zero21、FiSCoRe21。DECS第26–27页嵌入案例文字提取异常，补充逐页视觉阅读。没有运行模型训练、推理benchmark、作者API实验或候选代码。
- 全部22张原图裁切完成视觉检查。修正10处裁切边界，包括表格最后数据行、图例、坐标轴和图注；细项见pdf-check.json。实验数据未重绘或生成。
- 初始和发布前检查18份历史manifest、90条索引，读取全方向全日期内容，以arXiv去版本ID、DOI、规范URL和Unicode/标点/大小写归一标题及显式别名去重；无冲突。Speculative Actions两个标题手工合并，另一篇同名MemAgent未混入。
- CCF官方有效72页目录，物理57页ICLR A、58页EMNLP B已视觉核验；4篇ICLR逐篇oral/poster页、FiSCoRe主会论文集及官方链接日程668-MAIN均已核验。目标4A+1B、3oral+2poster，spotlight0，未知0。
- OpenAlex五份原始快照，查询2026-11-22，引用数264、5、29、238、9。FiSCoRe DOI按官方条目无/v1；没有用star/浏览量代替引用。
- 五个作者代码仓库固定提交，核读具名关键路径；DECS另读正式补充ZIP代码相关部分。配置、奖励、分组与论文差异如实写入正文和code-reading.json，非完整仓库安全审计。

- 全文 PDF 由同一份 Markdown 生成。逐项核对 233 条正文、表格和伪代码文本，缺失 0。
- 9 个独立公式与 111 处行内公式通过渲染检查。Markdown 采用 GitHub 数学语法。PDF 使用矢量独立公式与按基线对齐的高分辨率行内公式。对应 TeX、尺寸与哈希见 math-manifest.json。
- 22 张原论文图表的尺寸与解码 RGB 像素逐一匹配原图。正文、图像与数学路径的边界均通过检查。
- 全部 28 页由 PDFium 及 Poppler 渲染。逐页检查可读性、图表完整性及页码，再用 Poppler 全尺寸检查公式密集页 3、8、12、13、18、23。渲染记录及各页哈希见 pdf-check.json。
- 实际绘制的文字使用嵌入的 TrueType 字体与 ToUnicode 映射。字体字符覆盖通过，缺字 0。字体记录见 pdffonts.txt。
- 仓库的 20 项索引及公式排版测试通过。

全文 Markdown SHA-256 `f02f3c8d78d1f84a542f96488612964b43d44643597d253490f5db325bc0aeb4`。

PDF SHA-256 `6a4f452169179520e943e58562958fcfec10b9bec969bb40b2dbc3e2d54c0299`。
