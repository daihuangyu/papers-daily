# 原文与原图来源

核查日：2026-11-23。全部图片由指定v1原PDF以2×渲染裁切，未重绘数据，版权归作者/发布者。坐标为整页左上右下0—1比例。本文分别解释原图条件、基线、读法；OPST公式截图保留原式以说明一致性问题。原PDF与日期/项目页快照在工作目录留存，哈希见manifest。

| 文件 | 原图 / PDF页 | 来源 | 裁切坐标 |
| --- | --- | --- | --- |
| [o2-fig1.png](o2-fig1.png) | Figure 1 / 2 | [原PDF](https://arxiv.org/pdf/2611.07731v1) | [0.08, 0.08, 0.92, 0.335] |
| [o2-table2.png](o2-table2.png) | Table 2 / 7 | [原PDF](https://arxiv.org/pdf/2611.07731v1) | [0.07, 0.45, 0.94, 0.95] |
| [o2-fig4.png](o2-fig4.png) | Figure 4 / 8 | [原PDF](https://arxiv.org/pdf/2611.07731v1) | [0.08, 0.45, 0.92, 0.67] |
| [apd-fig1.png](apd-fig1.png) | Figure 1 / 3 | [原PDF](https://arxiv.org/pdf/2611.00852v1) | [0.06, 0.07, 0.94, 0.447] |
| [apd-table1.png](apd-table1.png) | Table 1 / 6 | [原PDF](https://arxiv.org/pdf/2611.00852v1) | [0.06, 0.06, 0.94, 0.419] |
| [apd-tables2-3.png](apd-tables2-3.png) | Tables 2–3 / 7 | [原PDF](https://arxiv.org/pdf/2611.00852v1) | [0.06, 0.06, 0.51, 0.57] |
| [gen-fig8-9.png](gen-fig8-9.png) | Figures 8–9 / 9 | [原PDF](https://arxiv.org/pdf/2611.05609v1) | [0.06, 0.06, 0.94, 0.455] |
| [gen-fig4.png](gen-fig4.png) | Figure 4 / 7 | [原PDF](https://arxiv.org/pdf/2611.05609v1) | [0.06, 0.06, 0.94, 0.258] |
| [gen-fig11.png](gen-fig11.png) | Figure 11 / 11 | [原PDF](https://arxiv.org/pdf/2611.05609v1) | [0.06, 0.25, 0.94, 0.455] |
| [opst-fig1.png](opst-fig1.png) | Figure 1 / 2 | [原PDF](https://arxiv.org/pdf/2610.32770v1) | [0.07, 0.07, 0.93, 0.27] |
| [opst-table1.png](opst-table1.png) | Table 1 / 7 | [原PDF](https://arxiv.org/pdf/2610.32770v1) | [0.08, 0.105, 0.94, 0.39] |
| [opst-formulas.png](opst-formulas.png) | Equations 5–6 and mixture definition / 5 | [原PDF](https://arxiv.org/pdf/2610.32770v1) | [0.08, 0.51, 0.92, 0.8] |
| [mp-fig2.png](mp-fig2.png) | Figure 2 / 3 | [原PDF](https://arxiv.org/pdf/2610.36477v1) | [0.08, 0.07, 0.92, 0.325] |
| [mp-table1.png](mp-table1.png) | Table 1 / 6 | [原PDF](https://arxiv.org/pdf/2610.36477v1) | [0.06, 0.06, 0.94, 0.485] |
| [mp-fig4-table3.png](mp-fig4-table3.png) | Figure 4 and Table 3 / 8 | [原PDF](https://arxiv.org/pdf/2610.36477v1) | [0.06, 0.06, 0.94, 0.245] |

## 代码核查

- O²PD: Overlap-Gated On-Policy Distillation on Reliable Prefix：No dedicated official runnable training implementation verified in original PDF links, author resources, metadata and exact-title/method searches on 2026-11-23; not a claim of nonexistence
- Adaptive-Policy Distillation: Knowledge Injection under Information Asymmetry：No dedicated official runnable training implementation verified in original PDF links, author resources, metadata and exact-title/method searches on 2026-11-23; not a claim of nonexistence
- On the Generalization Limitation of On-Policy Distillation for Post-RL Consolidation：Author project GitHub link is website source, not a verified dedicated training implementation
- On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models：No dedicated official runnable training implementation verified in original PDF links, author resources, metadata and exact-title/method searches on 2026-11-23; not a claim of nonexistence
- Proposing Better Rollouts for On-Policy Distillation：Author-linked overview repository only: README and one PNG; README says code is being prepared; no runnable training code or weights

MPOPD固定提交：`ac0971a1808d2719167be21c67501b66bed5a482`，README SHA-256：`0b499559dd9f8ea7d125e59ac244d3613f786b9591d9d1ea64f92801fa4066e3`。项目网站与README仅作辅助资料，不计论文名额。未运行GPU训练。
