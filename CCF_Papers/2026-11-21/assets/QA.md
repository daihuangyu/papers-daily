# 全文 PDF 与来源检查记录

日期：2026-11-21。实际使用用户授权的 macOS 本地环境与隔离发布检出。

- 完整阅读五篇论文与全部附录，共122页（AuxDPO主文11+独立补充10，LoongRL18，REPO30，RLCR38，ERPO15）。没有执行模型训练、推理benchmark或候选代码。
- 同一Markdown直接生成全文PDF，共29页、31597个Markdown字符。277个可见正文/公式/表格/代码文本项全部匹配，缺失0，文字边界异常0，没有短版替代。
- 全部29页已渲染，通过5张接触表逐页视觉检查；第4、14、19、25页另作全尺寸检查。未见缺字、重叠、截断、空白页。图片与图注保持整块导致部分页留白，正文没有省略。
- 20张原图裁切全部视觉检查；修正过AuxDPO图注下沿、LoongRL组合图后的残余文字、ERPO表9图注左沿。发布版包含完整表头、数据行、图例与原图注。
- 最终PDF中20张图解码后的RGB像素与assets原图逐一完全一致；图像尺寸与页面边界通过。匹配页码见pdf-check.json。
- 字体全字符覆盖检查通过。实际文字绘制字体均嵌入TrueType并带ToUnicode；Helvetica只是未使用资源。grad_、^T和“优于”替代不被字体覆盖的原记号，并在正文定义。
- 初始去重为13份历史manifest/65条索引；并发任务完成后重新读取15份历史manifest/75条记录以及sources.json，5篇无冲突。原arXiv与会议版手工合并，首日FWS排除。
- 10项索引回归测试通过。整合最新OPSD和VLM后，索引生成器验证16份报告、80个唯一来源、471个别名。只发布本期目录与3个必要索引。
- 官方证据包括CCF当前最终目录物理57/58页、4篇ICLR单独oral/poster页、ERPO主会论文集及官方链接日程6640-MAIN。最终4A+1B、3oral+2poster，spotlight0。
- OpenAlex原始快照5份，查询日期2026-11-21，引用数3、26、13、116、0。未用点赞/浏览/star替代。ERPO索引会议DOI未独立证实，保留为未核实字段。
- 4个论文官方代码仓库及KeyChain数据仓库固定提交核读；REPO仅取得论文第29页参考代码，完整训练仓库未找到。论文/实现差异见全文及code-reading.json。
- 运行中发现并保留OPSD提交af2d3a2、VLM提交11e592d；隔离发布基点为11e592de88c49a4f4d484599b40724ec54eaa734。原共享工作区其他方向文件没有由本任务改写。

## 字体工具输出

```text
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
Helvetica                            Type 1            WinAnsi          no  no  no       2  0
AAAAAA+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes     97  0
AAAAAB+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    101  0
AAAAAC+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    105  0
AAAAAD+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    109  0
```

生成使用ReportLab；渲染使用pypdfium2；全文与原图像素核对用pypdf；边界检查用pdfplumber。工作区独立fontconfig不改系统字体。

Markdown SHA-256：c7acad5cc3667e5ae67a61c125cc1f597332f3130d4dc9e74f1c62e0ad19bfae

PDF SHA-256：bf60009ff7ed33a2a9dec8aa8f2dec01b805be490eb791b983b7b5cb8f557b2c

实际推送后另行回读远端commit、全部本次文件blob与其他方向tree，回执在成功后保存；此文件不预先声称推送成功。
