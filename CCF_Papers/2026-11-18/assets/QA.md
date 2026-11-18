# 制品检查记录

检查日期：2026-11-18。实际环境为获授权的macOS本地工作区；未声称已在Linux云端执行。

- 五篇完整源PDF及附录共138页已阅读；正文区分原文事实、本文推导与未披露项。
- 最终报告为25页，Markdown共33093字符。265个可见文本行/段（包括公式、表格和代码）逐一在PDF提取文本中匹配，缺失0。
- 19张真实原始图表已保留并逐一检查；图源物理页码、裁切框、源PDF及PNG哈希见figures.json。无重画实验结果。
- 最终PDF全部25页已按1.5倍渲染并检查；公式密集页另作原尺寸检查。无缺字、越界、叠字、空白页，图注与图像保持同页，标题与紧随的原子图/代码块保持同页。
- 标题别名、arXiv版本、DOI、URL、全仓跨日/跨方向去重检查通过。此前6份manifest、30条来源，本期5条均无冲突。
- 10项索引回归测试通过。已覆盖CCF录用年度（允许更早预印本）、正式长文/证据要求、spotlight不混为oral、引用来源日期以及改题查重。
- 本次没有运行训练、推理benchmark或论文复现实验；报告中的模型结果来自原文。
- 字体使用现有转换后可嵌入TrueType中文字体，实际名称STSongti-TC-Bold；所用文字字体均嵌入且有Unicode映射。默认Helvetica为未使用的资源。

## 字体工具记录

```text
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
Helvetica                            Type 1            WinAnsi          no  no  no       2  0
AAAAAA+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    111  0
AAAAAB+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    115  0
AAAAAC+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    119  0
AAAAAD+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    123  0
AAAAAE+STSongti-TC-Bold              TrueType          WinAnsi          yes yes yes    127  0
```

原生Poppler最初缺少可写fontconfig缓存，改用本工作区独立配置后完成检查；没有改系统字体或用户配置。PDF逐页渲染由pypdfium2完成，边界检查使用pdfplumber。

## 最终文件SHA-256

Markdown：91c4e0391cb9746911cbb27baef90d4f09e6ad7ae8979478ea35ab28d6185fe2

PDF：ddb3bb1fb593d7cf35f36e0a2a65afde0ee7958793244a5da8682574eee83a01

详细可机读检查见pdf-check.json。发布后的远端提交及逐文件blob核对结果另保存于交付回执，不在提交前预先宣称发布成功。
