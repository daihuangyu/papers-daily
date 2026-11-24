# PDF 全文与公式检查

报告日期 2026-11-23。2026-11-24 发布前重新排版并检查全部 24 页。

- 全文阅读五篇正式主会 PDF 及附录，共164页：OpenThoughts72、LongWriter-Zero26、ThinKV35、AceReason19、SFT程度12。相关原图、完整图例和数据表另经视觉阅读。未执行模型训练、推理benchmark、作者代码或生成解答。
- 20张原图在4张图源接触表逐一检查，修正了旋转表头、最右列、mask箭头、坐标轴、面板名和表头边界。最终ThinKV Figure8的完整原图注在报告第13页再次确认。没有重绘或生成实验图。
- 起始18份manifest/90条索引；获取当天VLM/OPSD两笔并发提交后，重新读取全部20份manifest/100条索引，另对未发布的11月22日5篇去重。arXiv去版本、DOI、规范URL、Unicode/标点/大小写归一标题和别名无冲突。
- CCF2026正式72页目录的物理57/58页已视觉确认ICLR A/EMNLP B。四篇ICLR逐篇官方oral/poster条目；EMNLP主会main.128和官方日程340-MAIN匹配。组成4A+1B、3oral+2poster，无spotlight冒称oral。
- OpenAlex五份原始快照，查询日2026-11-23，数值438、32、25、74、0。SFT论文精确首发日未知，官方只有2026-10，不采索引占位日期2026-01-01。
- 三个GitHub固定提交和AceReason官方固定HuggingFace评测归档做具名路径静态阅读；ThinKV只核得论文伪代码，Code标签未链接实现。代码/论文与脚本参数差异见正文及code-reading.json；未把评测发布等同训练完全公开。

- 全文 PDF 由同一份 Markdown 生成。逐项核对 262 条正文、表格和伪代码文本，缺失 0。
- 18 个独立公式与 111 处行内公式通过渲染检查。Markdown 采用 GitHub 数学语法。PDF 使用矢量独立公式与按基线对齐的高分辨率行内公式。对应 TeX、尺寸与哈希见 math-manifest.json。
- 20 张原论文图表的尺寸与解码 RGB 像素逐一匹配原图。正文、图像与数学路径的边界均通过检查。
- 全部 24 页由 PDFium 及 Poppler 渲染。逐页检查可读性、图表完整性及页码，再用 Poppler 全尺寸检查公式密集页 3、8、12、20。渲染记录及各页哈希见 pdf-check.json。
- 实际绘制的文字使用嵌入的 TrueType 字体与 ToUnicode 映射。字体字符覆盖通过，缺字 0。字体记录见 pdffonts.txt。
- 仓库的 20 项索引及公式排版测试通过。

全文 Markdown SHA-256 `cac962eab88ae3cfcb08b421a5da793b35d45d86ce098c2c230bf6c95d949e72`。

PDF SHA-256 `904fd17e3f73b5da12068fcc3cfcbd6f12c202b2e44fde5d0a03398862c8b21f`。
