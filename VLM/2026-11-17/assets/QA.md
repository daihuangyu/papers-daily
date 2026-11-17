# 2026-11-17 检查记录

- 主论文5篇，均核对arXiv v1日期；窗口2026-10-17至2026-11-17。
- update_index.py校验3份日报、15个唯一来源、60个去重别名；包括VLM和OPSD历史，未重复。
- report.md约15,692字符；同一完整Markdown渲染report.pdf，13页、10个原图/表嵌入对象，无空白页。
- 最新PDF逐页缩略图与原图已检查，无图表裁断、文字重叠或缺字；正文嵌入可映射Unicode的中文TrueType子集。本次本地补位使用macOS Songti，未改动既有云端Linux Noto字体模块。
- 原表1的BlinkOPD 9B行已纳入裁图；原图和来源表一一对应。公式使用可直接读的纯文本记号，原始数学排版保留于论文原图/PDF链接。
- PDF SHA256：170f72ddf04a88701cc71ebeb70b3caa1c628f3f8443d6ca0f9f618f322fcded。
- 未训练任何模型；未把作者结果当作本次复现实验。Safety as a Budget摘要与Table 1的over-refusal冲突已明确保留。
- FlowShift项目页地址出现在作者论文中，但本次网页访问失败；可用性未确认，不编造代码仓库。
- 云端迁移仍未完成：云端GitHub写入与Linux PDF已验证，托管日程因环境绑定待登录核验而暂停。当前报告由旧本地日程补位，不宣称关机也能执行。
