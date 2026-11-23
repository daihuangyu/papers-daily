# 每日论文云端运行约定

## 任务与授权

- 公开仓库：https://github.com/daihuangyu/papers-daily，分支main。用户于2026-11-18要求改为Public，已在GitHub设置页核验生效。用户已明确授权每日自动写入本仓库，无需每次等待人工确认。
- 云端计划：Asia/Shanghai每天09:00，首个日程2026-11-17 09:00，长期运行，无结束日期。由ChatGPT云端定时任务执行研究、写作、PDF生成和发布，不是创建普通聊天，也不依赖用户电脑开机。
- 迁移状态（2026-11-16）：**尚未完成，等待云端执行环境绑定核验**。定时服务确实创建了持久任务并返回正确日程和next_run_time=2026-11-17T09:00:00+08:00，但环境元数据仅为environment_selection=inherited、environments=null；接口说明null表示继承详情不可用，不能据此证明脱离本机。公开的create/update工具没有环境选择参数。网页核验停在ChatGPT登录页。
- 新任务已暂停（is_enabled=false），避免把未验证的环境当作迁移完成；保留完整任务指令与日程，确认显式云端环境后再启用。此处不将它标记为已完成迁移，也不影响旧本地vllm或原有已暂停跟踪任务。
- GitHub云端写入已通过实际提交验证：提交cb89908948dc73c98c2c2dfc3467b632adaf9ba9已在main读回；UTF-8运行约定与263425字节PDF二进制上传/读回一致，PDF blob SHA为d760fa9c5218f045760cd9d6e259230b7c5e8b67。采用单父提交与force=false的fast-forward更新。首日2026-11-16目录的15个tree/blob条目SHA全部保留。
- 创建前已检查云端任务：没有相同的papers-daily每日入库任务；另有已暂停的Vision-OPD论文跟踪任务，保持不变。
- 原本地任务ID为`vllm`，由原聊天在云端迁移验证后停用。本流程不创建或更改本地任务。
- 持久状态以远端仓库为准。每次运行重新读取AUTOMATION.md、scripts/、sources.json和全部方向的manifest.json；不得假设上一次临时工作区仍存在。
- 只使用云端GitHub连接与云端临时工作区；禁止依赖个人电脑路径、SSH、浏览器登录凭据、从本机复制token/私钥。连接失败需报告具体阻塞。

## 选题、时间窗与去重

- 优先小型VLM/MLLM训练与后训练；参考Vision-OPD和MMSearch-R1方向，覆盖on-policy蒸馏、SFT、RL/GRPO、偏好优化、细粒度视觉感知、视觉推理、多模态搜索/工具agent。
- 每次以Asia/Shanghai当地日期D归档到`VLM/D/`；新论文以首次公开日期为准，时间窗为D的上一个日历月同日至D（短月夹到月末）。例如2026-11-17的窗口是2026-10-17至2026-11-17。核实arXiv v1日期，不能以新版日期把老论文当新论文。
- 目标5篇不同的新论文。先检索并核验一手论文页面/PDF，再与sources.json和全部历史manifest交叉查重：arXiv去版本ID、DOI、规范URL、Unicode/大小写/标点归一化标题；人工复核改题、预印本与会议版、近似标题是否同篇。版本更新不默认算新论文。
- 不足5篇时报告实际数量、检索范围和不足原因；可以为0篇。禁止用旧论文、博客、教程、方向锚点或一篇拆多篇凑数。
- 博客与基础教程只是辅助材料，放在各篇延伸阅读或manifest的auxiliary_sources，不列入sources主清单。
- 2026-11-16已有5篇vLLM工程文章是用户确认的首日例外，保留现有Markdown、PDF、manifest和assets原样；迁移当天不另生成5篇。

## 每篇内容与图片

中文写作，逐篇提供：问题及动机；方法机制与训练流程；公式逐符号解释与关键推导；小白直觉和可手算数值例子；训练loss/reward及各项含义；数据来源、过滤、构造和采样；模型规模、teacher/student设置、硬件、训练步数/时长/显存/GPU-hours及成本。论文未披露的内容明确标为未披露，不编造数字。

给出实现要点、伪代码、关键超参数；核心实验、消融、所用baseline及可比条件（初始化、数据、算力、继续训练或从头训练、公平性）；局限、失效场景、复现步骤与检查项。明确区分论文事实、作者实测、本文推导与建议。

引用论文原架构图、流程图及关键结果图，下载真实原图或从论文PDF裁图，保存assets/。每幅注明原论文图号、标题、论文URL、页码/图源及必要的访问日期，逐图解释模块、箭头、坐标轴、基线、数值与结论适用条件。不得生成或重画伪实验结果。若原图获取失败，逐项说明缺失，不能用虚构图片替代；辅助教学图必须明确标注。

提供论文/官方项目/代码、官方或作者博客（如存在）及按难度分级的基础教程链接；没有作者博客就如实说明。辅助材料不计入5篇。

## 目录、清单和索引

每日报告目录含`report.md`、从最终Markdown全文生成的同内容`report.pdf`、`manifest.json`和`assets/`（含SOURCES.md图源表）。其他方向以后使用独立一级目录，不混入VLM。

manifest保持现有date、direction、topic、window_start、window_end、sources结构；新增报告还应有target_count=5、actual_count、shortfall_reason（不足时必填）、search_log（检索词/日期/剔除原因）、auxiliary_sources及PDF检查记录。sources中的每项包括：

- id、type=`paper`、title、published（首次公开日期）、url；
- arxiv_id（不带vN）、doi（如有）、canonical_urls（同篇的论文规范地址，不能放共享代码库或图源）；
- publication_evidence_url、pdf_url、code_url（如有），以及原图/页码的来源记录。

运行scripts/update_index.py重建根README、所有已有方向的README和sources.json v2；v2包含来源记录、别名及归档位置。不重复收录同篇的不同版本或会议版。模糊标题仍需人工复核。

## Linux PDF与排版验证

OPSD 方向自 2026-11-23 排版修订起，正文公式使用原生 LaTeX 行内 `$...$` 与独立 `$$` 数学块，禁止继续用未排版的线性字符式代替主要公式。GitHub 原生渲染同一源码；PDF 由 MathJax 本地生成公式素材并嵌入。先执行 `npm ci --prefix scripts/mathjax` 准备固定版本依赖，具体约定见 [OPSD数学排版](OPSD/FORMATTING.md)。该规则只更新公式排版，不改变其他方向内容、选题、时间窗口或发布授权。

Python依赖见requirements.txt；Linux系统需`fonts-noto-cjk`和`fonts-dejavu-core`，可用Poppler检查输出。scripts/cjk_font.py从Linux Noto Sans CJK SC取字形，转换为可嵌入TrueType子集，数学补充字形来自DejaVu Sans，不引用macOS Songti。可用PAPERS_CJK_FONT及PAPERS_CJK_FONT_INDEX指定其他可嵌入中文字体；缺字直接报错。

```sh
python3 scripts/build_report.py VLM/YYYY-MM-DD/report.md
python3 -m unittest discover -s scripts/tests -v
python3 scripts/update_index.py
pdffonts VLM/YYYY-MM-DD/report.pdf
pdftoppm -scale-to 1400 -png VLM/YYYY-MM-DD/report.pdf /tmp/papers-daily-page
```

必须逐页检查PDF的中文、公式、表格、代码、图片和图注；确认无缺字、截断、重叠或空白页，正文与Markdown相同，不能用短版替代。当前renderer支持Markdown标题、段落、表格、代码块和本地图片；复杂LaTeX公式需先渲染成assets公式图并插入同一Markdown，在PDF中保持相同公式，不把未渲染LaTeX当成排版完成。

`scripts/tests/fixtures/cloud-font-smoke.md/.pdf`是迁移验证样例，不是每日论文。已在Linux生成并渲染检查：中文、α/β/μ、转置符号、代码、表格和仓库原图可读；pdffonts显示PapersDailyCJK-Regular的emb/sub/uni均为yes。6项去重和时间窗测试通过。

## 云端GitHub发布

优先使用已授权GitHub连接，完成与普通commit/push等价的单父提交及fast-forward更新main，禁止force push：

1. 读取main当前head与完整递归tree，保留base commit/tree及所有非本次改动。
2. 在云端仅获取需要的文本和素材；GitHub contents接口对大于1MB的旧二进制可能返回空content，不能把空内容当作真实文件。原有PDF/大图保留其远端blob引用，无需重新下载或提交。完整远端tree可存临时JSON，索引校验用`python3 scripts/update_index.py --remote-tree /tmp/remote-tree.json`检查未改动远端文件的存在性。
3. 生成并检查本次新报告的全部本地文件。新文件必须真实生成，不得以远端inventory替代本次PDF与图片检查。
4. 通过GitHub create_blob上传文本（utf-8）与PDF/图片（base64），检查返回SHA；二进制编码在代码中处理，按需分片读取，勿把base64打印给用户。
5. create_tree以刚读取的base_tree_sha为底，只增加/替换本次文件；create_commit以刚读取的main head为唯一parent；update_ref设置branch_name=main、expected_sha=旧head、force=false。
6. 若head已变，重新读取并合并到最新树后重做，保护其他方向内容，不覆盖并发提交。不得force或改为本机SSH推送。
7. 成功后重新读取main与新增文件，核对提交、文件SHA与清单条目，返回当天篇数、Markdown/PDF链接、commit及不足原因。失败明确报告具体阶段，不能声称已发布。

scripts/publish.sh仅供已有授权HTTPS origin的托管云端checkout使用；它不配置凭据，拒绝SSH和其他仓库地址。没有云端git凭据时使用上述连接提交路径，不复制本机凭据。

如果当天已存在通过验证的完整报告，按幂等方式核验并返回原链接，不重复生成或重复选篇。若上次失败，优先恢复该日缺失步骤。任何GitHub连接、检索、原图、字体、PDF或提交权限问题都要精确记录并通知用户，不得把普通云端聊天或只写了Markdown当作完成。
