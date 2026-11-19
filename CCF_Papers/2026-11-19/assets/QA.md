# 全文 PDF 与来源检查记录

日期：2026-11-19。实际执行于用户授权的 macOS 本地工作区，不宣称 Linux 云端执行。

- 完整阅读五篇 PDF 及附录，共 215 页。模型实验结果均来自论文，本次没有运行训练、推理 benchmark 或复现实验。
- 最终报告 22 页，Markdown 28156 字符，逐行/段核对 251 个可见文本项（含公式、表格、代码），未匹配项 0。未用短版替代全文。
- 原始论文图表裁切 18 张；图号、版本、物理页、裁切坐标和哈希见 figures.json。修正过 TROLL 顶部、GEPA 右侧及 GAC 表尾边界，最终无数据行/坐标标签丢失。
- 全部 22 页渲染后逐页视觉检查；公式密集的第 10、15、18、19 页另作原尺寸检查，最终改动的第 21 页重新检查。没有缺字、边界截断、重叠、空白页或丢失图注。
- 18 张图从最终 PDF 解码后的 RGB 像素与 assets 原图逐一完全一致，图像坐标均在页面内容边界内；详细匹配和所在报告页见 pdf-check.json。
- 字体字符覆盖检查缺字 0；实际用于文字绘制的全部字体均嵌入 TrueType 且有 ToUnicode 映射。默认 Helvetica 仅为未使用资源。
- 阅读 sources.json 与全部 9 份历史 manifest，核对 45 条既有来源，本期 5 条均无跨日/跨方向冲突。RwS 与昨天 FWS 人工确认不同论文；GAC 预印本与正式论文合并身份。
- 10 项仓库索引回归测试通过，包括 CCF 年度、正式长文证据、展示类型、引用出处和题名别名规则。
- 引用快照来自 OpenAlex，查询日 2026-11-19；原始 JSON 已保留，未使用浏览量或 star。GAC 官方 Virtual Pres 原样记录，oral/poster 未确认。
- 核心代码阅读及固定提交见 code-reading.json；TROLL 训练代码未发布，GAC 为组件与接入模板，ELF 的论文/代码采样差异均在正文披露。

## 字体工具记录

```text
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
Helvetica                            Type 1            WinAnsi          no  no  no       2  0
AAAAAA+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    108  0
AAAAAB+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    112  0
AAAAAC+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    116  0
AAAAAD+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    120  0
```

PDF 生成使用 ReportLab，渲染使用 pypdfium2，文字/图像完整性使用 pypdf，边界使用 pdfplumber。字体工具使用工作区独立 fontconfig 配置，未修改系统字体设置。

## 最终文件 SHA-256

Markdown：17de79ec48d2ca567c9ee68b57cc6f17edf3271204125e0cbf93a5c1fa0729c3

PDF：64f6a9f1c475ff0159ba96b032bb1d165c6ca37b0269aff4f2ca34fbbbbcb1c0

发布后再核验远端 commit、全部本次变动文件 blob 与其他方向 tree；回执在实际推送后生成，不在此预先宣称发布成功。
