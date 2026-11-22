# 2026-11-22 全文与 PDF 检查

- 实际执行环境：本机 macOS 补位、既有授权 SSH；未复制凭据。已读取远端 AUTOMATION.md 与云端迁移聊天，云端长期任务仍暂停待环境绑定核验，没有同日已验证云端报告；未删除本地任务。
- 选题：5 篇不同论文，arXiv v1 日期为11月14日/19日，均在2026-10-22至2026-11-22。核对全部方向历史manifest与sources.json的arXiv/DOI/URL/归一化标题；旧文新版和教程不占名额。
- 最终正文：30,176个Unicode字符（含代码、公式、链接），24页PDF，18幅原论文图表。全部正文由同一Markdown渲染，没有短版PDF。
- PDF SHA-256：`de6901c6ea617f32f325753e9914500b8cc169c128d39c44187aeadd34bee43d`。
- 全文对照：pypdf提取文字，去除排版空白和实际HTTP Markdown链接语法后，所有正文、代码行及表格单元格匹配；保留公式中的方括号和条件符号。图片18/18均存在且嵌入PDF，未发现空白页。
- 字体：生成前逐字符检查覆盖；本机Songti TTC转嵌入TrueType，pdffonts显示五个实际中文/数学正文STSongti子集emb/sub/uni均为yes。ReportLab默认Helvetica资源不用于中文正文。偏好关系使用中文“优于”，没有保留缺字字形或未渲染LaTeX。
- 原图：从五份固定v1 PDF裁取，保留架构、表头、基线、数值/曲线、图例和必要原注；每图在正文标明图号、物理页并解释。裁图范围、来源、PNG尺寸与哈希见manifest.json和SOURCES.md。教学算例均为文字，未伪造成实验图。
- 逐页视觉检查：Poppler将全部24页渲染为长边1100像素PNG，逐页检查四张拼图及原图细节。中文、公式、代码缩进、表格、图注、标题、页码无截断/重叠/缺字；大图整体分页。两张窄结果表按内容缩放，避免不必要的巨大字号。
- 方法事实：SARGO两种策略角色、几何均值而非权重和守恒；FaithTwI受控/真实crop分母和4B/7B基线分开；EmbodyAgent二元筛组与shaped reward方差分开、训练保护从round1存在、调用减少但117→136秒；WorldAuditor仍需frontier reviewer/六份动作预算/论文列价；KLTDPO实际附录与实现缺失，DRO对偶仅标为本文推导。
- 代码核验：WorldAuditor作者仓库在GitHub connector及网页均为空，未称已发布实现；其他四篇未找到可验证官方完整方法代码。没有把第三方code标签、相邻论文仓库或通用依赖当本篇实现。
- 仓库检查：`python3 -m unittest discover -s scripts/tests -v`的10项全部通过。索引用scripts/update_index.py从已提交HEAD的隔离快照加本次VLM目录重建，保护其他方向尚未提交的内容；只提交本次文件及必要索引，正常SSH推送main，不使用force。

以上是文档和来源验证，不是模型实验复现。未披露超参数、硬件、成本和统计不确定性在正文保留。
