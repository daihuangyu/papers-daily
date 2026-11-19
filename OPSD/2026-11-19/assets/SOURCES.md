# 原图与原文来源

访问与核查日期：2026-11-19。以下均从对应版本原 PDF 以 2× 比例渲染后裁切；未重画、重标数值或拼接结果。坐标为左、上、右、下相对整页的 0—1 比例。原图版权归原作者/发布者。

| 文件 | 原文图表 / PDF页 | 版本来源 | 归一化裁切范围 |
| --- | --- | --- | --- |
| [swe-fig1.png](swe-fig1.png) | Figure 1 / 5 | [PDF](https://arxiv.org/pdf/2611.21020v1) | [0.11, 0.075, 0.89, 0.365] |
| [swe-table1.png](swe-table1.png) | Table 1 / 6 | [PDF](https://arxiv.org/pdf/2611.21020v1) | [0.11, 0.625, 0.89, 0.823] |
| [swe-fig4.png](swe-fig4.png) | Figure 4 / 10 | [PDF](https://arxiv.org/pdf/2611.21020v1) | [0.11, 0.615, 0.9, 0.785] |
| [swe-fig6.png](swe-fig6.png) | Figure 6 / 24 | [PDF](https://arxiv.org/pdf/2611.21020v1) | [0.11, 0.245, 0.89, 0.8] |
| [stream-fig1.png](stream-fig1.png) | Figure 1 / 7 | [PDF](https://arxiv.org/pdf/2611.22298v1) | [0.115, 0.25, 0.89, 0.66] |
| [stream-fig2.png](stream-fig2.png) | Figure 2 / 13 | [PDF](https://arxiv.org/pdf/2611.22298v1) | [0.115, 0.1, 0.89, 0.605] |
| [stream-fig5.png](stream-fig5.png) | Figure 5 / 22 | [PDF](https://arxiv.org/pdf/2611.22298v1) | [0.115, 0.3, 0.89, 0.77] |
| [stream-table10.png](stream-table10.png) | Table 10 / 25 | [PDF](https://arxiv.org/pdf/2611.22298v1) | [0.115, 0.19, 0.89, 0.46] |
| [agentic-fig2.png](agentic-fig2.png) | Figure 2 / 4 | [PDF](https://arxiv.org/pdf/2611.12799v2) | [0.115, 0.1, 0.89, 0.495] |
| [agentic-table1.png](agentic-table1.png) | Table 1 / 6 | [PDF](https://arxiv.org/pdf/2611.12799v2) | [0.165, 0.1, 0.835, 0.592] |
| [agentic-table2.png](agentic-table2.png) | Table 2 / 8 | [PDF](https://arxiv.org/pdf/2611.12799v2) | [0.535, 0.56, 0.91, 0.88] |
| [react-fig1.png](react-fig1.png) | Figure 1 / 2 | [PDF](https://arxiv.org/pdf/2611.19244v1) | [0.53, 0.105, 0.92, 0.425] |
| [react-fig2.png](react-fig2.png) | Figure 2 / 4 | [PDF](https://arxiv.org/pdf/2611.19244v1) | [0.11, 0.04, 0.9, 0.335] |
| [react-fig4.png](react-fig4.png) | Figure 4 / 8 | [PDF](https://arxiv.org/pdf/2611.19244v1) | [0.11, 0.045, 0.9, 0.35] |
| [react-table5.png](react-table5.png) | Table 5 / 9 | [PDF](https://arxiv.org/pdf/2611.19244v1) | [0.27, 0.25, 0.865, 0.405] |
| [dynamics-fig1.png](dynamics-fig1.png) | Figure 1 / 2 | [PDF](https://arxiv.org/pdf/2611.08016v1) | [0.115, 0.06, 0.89, 0.4] |
| [dynamics-fig4.png](dynamics-fig4.png) | Figure 4 / 7 | [PDF](https://arxiv.org/pdf/2611.08016v1) | [0.115, 0.04, 0.89, 0.34] |
| [dynamics-fig5.png](dynamics-fig5.png) | Figure 5 / 9 | [PDF](https://arxiv.org/pdf/2611.08016v1) | [0.165, 0.1, 0.835, 0.365] |
| [dynamics-fig8.png](dynamics-fig8.png) | Figure 8 / 11 | [PDF](https://arxiv.org/pdf/2611.08016v1) | [0.165, 0.38, 0.835, 0.825] |

各 PNG SHA-256 及源 PDF SHA-256 见上级 manifest.json；源 PDF 留在本次研究工作目录，未重复上传整篇论文。所有图表在 report.md 内有来源、条件、基线与读图说明。

## 公开代码核查

仅 SWE-OPD 核验到专门作者实现；固定提交 0884a04f4c9e2b4baaa47261b2016d425ac3f63e。检查文件与 SHA-256：

- [readme.md](https://github.com/ernie-research/SWE-OPD/blob/0884a04f4c9e2b4baaa47261b2016d425ac3f63e/readme.md)：`cf784e662f33509e8add319d8a3920a91de9e20711de99769996db4eca7c2da2`
- [offline_opd/slime/examples/coding_agent_opd/scripts/offline_opd/OFFLINE_OPD_FORMULAS.md](https://github.com/ernie-research/SWE-OPD/blob/0884a04f4c9e2b4baaa47261b2016d425ac3f63e/offline_opd/slime/examples/coding_agent_opd/scripts/offline_opd/OFFLINE_OPD_FORMULAS.md)：`69a8dab0b4e0b2d6894f3229dca34e7317c2049d28dbaf67e4236fa16399d546`
- [offline_opd/slime/slime/backends/megatron_utils/loss.py](https://github.com/ernie-research/SWE-OPD/blob/0884a04f4c9e2b4baaa47261b2016d425ac3f63e/offline_opd/slime/slime/backends/megatron_utils/loss.py)：`f22add3f81286a86a490cf097cb1c59cd552ea7cd8cd48de92176ef66c5d5ce8`
- [offline_opd/slime/slime/rollout/on_policy_distillation.py](https://github.com/ernie-research/SWE-OPD/blob/0884a04f4c9e2b4baaa47261b2016d425ac3f63e/offline_opd/slime/slime/rollout/on_policy_distillation.py)：`0e39006eae197d198946904930d66ecffc3753447d2347a0bc74cf02f50c3ceb`
- [vanilla_opd/README.md](https://github.com/ernie-research/SWE-OPD/blob/0884a04f4c9e2b4baaa47261b2016d425ac3f63e/vanilla_opd/README.md)：`61a60b799215ed8a4d6f32c1d0b2a57d3c8c5e82d64e628c11e7a25276fcddd1`

代码未运行训练；内部 sandbox 依赖、论文与可微归一化差异及 selector 差异见正文。其他四篇未核验到作者方法代码，基础环境/框架库不冒称方法实现。
