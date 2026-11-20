# 全文 PDF 与来源检查记录

日期：2026-11-20。实际执行于用户授权的 macOS 本地工作区。

- 完整阅读五篇 PDF 及附录，共 156 页；本次没有运行论文模型训练、推理 benchmark 或复现实验。
- 最终报告 25 页，Markdown 30835 字符，逐行/段核对 266 个可见文本项，含公式、表格、代码，未匹配项 0。全文 PDF 直接从同一 Markdown 生成，没有短版替代。
- 原始论文图表裁切 19 张；版本、图号、物理页、裁切坐标和哈希见 figures.json。修正过 ScaleRL 图顶部与图注边界、Global Resolution 表头、RLVR 图注底部与 STORM 图注边界，最终没有数据行、标题或标签缺失。
- 全部 25 页已渲染并通过全部 5 张接触表逐页视觉检查；公式/图表密集的第 10、15、20、22 页另作全尺寸检查。没有缺字、内容重叠、边界截断或空白页。部分页因图与图注保持整块而留白，非正文缺失。
- 全部 19 张图从最终 PDF 解码的 RGB 像素与 assets 原图逐一完全一致，图像边界检查通过；匹配和报告页码见 pdf-check.json。
- 字体字符覆盖缺字 0；实际用于文字绘制的全部字体嵌入 TrueType 且有 ToUnicode。Helvetica 仅为未使用资源。
- sources.json 与全部 12 份历史 manifest 的 60 条来源已去重。Global Resolution 改题、STORM 会议版/预印本身份已人工合并；首日 FWS 不重收。
- 10 项仓库索引回归测试通过；索引生成器验证 13 份报告、65 个唯一来源、377 个别名。
- OpenAlex 五条原始引用快照保留，查询日 2026-11-20，数值 115、3、3、326、0，未用浏览量或 star。
- CCF 官方最终目录第 57、58 页已视觉核查；ICLR 四篇逐一用官方 oral/poster 页确认，STORM 用主会论文集及官方链接日程确认 poster。组成 4A+1B、3oral+2poster。
- 三个官方代码仓库固定提交和核读路径见 code-reading.json。ScaleRL 仅公开曲线拟合；Global Resolution、RLVR 自有完整实现未找到；STORM 论文/公开代码差异在正文说明。

## 字体工具输出

```text
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
Helvetica                            Type 1            WinAnsi          no  no  no       2  0
AAAAAA+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    108  0
AAAAAB+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    112  0
AAAAAC+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    116  0
AAAAAD+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    120  0
```

生成使用 ReportLab；逐页渲染使用 pypdfium2；全文与图像完整性使用 pypdf；文字和图像边界使用 pdfplumber。使用工作区独立 fontconfig，未修改系统字体设置。

## 文件 SHA-256

Markdown：af36a75aff168d4b24ee5837295f2533065337b24fbbfbeceecd99fcf7db9fe0

PDF：ffa4871d9baef04c73df6840251b6e73537c34bd62ea49aa2218d6989fdc4e96

发布后另行回读远端 commit、全部本次文件 blob 及其他方向 tree。回执在实际推送后保存；此记录不预先声称推送成功。
