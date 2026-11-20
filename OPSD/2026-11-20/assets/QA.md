# 2026-11-20 全文、原图与 PDF 检查

检查日期：2026-11-20，Asia/Shanghai。本次完成论文精读、静态源码核对和文档验证；没有执行论文模型训练。

- 选题与去重：5篇首次公开在2026-10-20至2026-11-20内的新论文。已对sources.json的55条历史来源、全部11份历史manifest交叉检查去版本arXiv、DOI、规范URL、NFKC标题与别名，并人工核对版本/改题身份。包含并发完成的VLM/2026-11-20，未与其他方向重复。
- 原文：下载指定版本的5份原PDF，共72页，读取与本文有关的方法、核心实验、训练条件、消融和附录；页数不是“逐页复现全部证明/参考文献”的声明。B-OPSD/SD-OPD读v2，发布日期仍按v1计；Proxy OPD旧稿的11月版本号变化未当作新论文。
- 图源：20幅原架构/机制图与实验图表，源PDF以2×渲染后裁切。检查了源页、裁图预览和最终PDF，保留图形、坐标轴、图例、表格及相应原图注；只裁切与缩放，没有重绘或改实验数值。SOURCES.md和manifest记录版本、页码、坐标与SHA-256。
- 内容：report.md共32,266字符；report.pdf从同一全文生成。261条非空正文、公式、伪代码与表格行去掉Markdown标记和链接目标、归一化空白后逐行核对，缺失0条。使用可读Unicode/ASCII数学式，没有把未渲染LaTeX当完成；center算子用花括号，避免被Markdown误当相对链接。
- 图片：PDF内20个原图对象，与assets中的20个PNG解码RGB像素哈希逐一匹配，无遗漏或替换。原PNG与源PDF哈希见manifest，机器结果见pdf-check.json。
- 字体：当前本地macOS使用已有可嵌入Songti TrueType字体；全部可见字符先做cmap检查，缺字0。4个STSongti-TC-Bold字体子集都嵌入FontFile2并有ToUnicode。没有声称本次运行用Linux Noto。grad、sg、τ_plus等记号已在正文定义或上下文解释。
- 渲染：25页A4，以1.5×逐页渲染并查看覆盖全部25页的contact sheets；另按完整渲染尺寸检查第7、12、17、19、21页的代码核查、证明、公式和密集结果表。无空白页、缺字、文字越界、图注截断或可见重叠；小表保持整块，O3PD结果汇总不再跨页拆成单行。
- 生成：复用仓库scripts/build_report.py，由本日报assets/render_report.py调整行距/段距、图片高度和小表KeepTogether；共享renderer未修改。可设置PAPERS_DAILY_HOST_FONT为可嵌入TrueType中文字体再运行该脚本，不设则使用仓库原字体流程。字体、renderer、adapter与全文哈希见pdf-check.json。
- 源码：B-OPSD固定8df0eae3424c9d03e82fa75346388e016a7dec05；SD-OPD固定ae19768776759d3788c4c5a65d5562f2b0929eba。已核对loss、打分/对齐、相关入口。B-OPSD的brief上限/重试、token均值与论文序列均值、数据路径差异明确披露；SD-OPD实际入口direct/current、未重归一化centering、零reward及clipping比较条件明确披露。没有把其他三篇的基础框架或第三方实现冒称作者代码。
- 基线口径：区分Avg@12/16与pass@12/16，评估采样seed与训练seed，逐benchmark最佳与同checkpoint最佳，等rollout/episode与等FLOPs/API成本。原文未披露的硬件、费用或训练超参数直接标缺失，没有填默认值。
- 仓库：现有scripts/tests的10项测试通过。索引由scripts/update_index.py重建；只提交本日报目录、OPSD/README.md、根README与sources.json。其他方向与AUTOMATION.md/共享scripts保持当前基线原样。SSH普通commit/push授权来自本次用户任务，禁止force push。

本记录与pdf-check.json是交付前验证；推送后的远端commit及全部新增文件将另行核验，结果在任务回执报告。
