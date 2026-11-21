# 原图、原文与代码来源

核查日期：2026-11-21。指定版本PDF以2×渲染后裁图，坐标为相对整页的左/上/右/下0—1比例；没有重绘、改写或生成实验数据。版权归原作者/发布者。源PDF在研究目录留存，不重复上传整篇论文。

| 文件 | 原图表 / PDF页 | 来源版本 | 裁切坐标 |
| --- | --- | --- | --- |
| [sh-fig1.png](sh-fig1.png) | Figure 1 / 1 | [PDF](https://arxiv.org/pdf/2611.20687v1) | [0.07, 0.23, 0.93, 0.59] |
| [sh-fig3.png](sh-fig3.png) | Figure 3 / 8 | [PDF](https://arxiv.org/pdf/2611.20687v1) | [0.1, 0.1, 0.9, 0.595] |
| [sh-fig4.png](sh-fig4.png) | Figure 4 / 14 | [PDF](https://arxiv.org/pdf/2611.20687v1) | [0.1, 0.105, 0.9, 0.55] |
| [bern-fig1.png](bern-fig1.png) | Figure 1 / 2 | [PDF](https://arxiv.org/pdf/2611.02093v1) | [0.055, 0.085, 0.95, 0.33] |
| [bern-fig3.png](bern-fig3.png) | Figure 3 / 4 | [PDF](https://arxiv.org/pdf/2611.02093v1) | [0.055, 0.085, 0.95, 0.355] |
| [bern-table2.png](bern-table2.png) | Table 2 / 6 | [PDF](https://arxiv.org/pdf/2611.02093v1) | [0.055, 0.075, 0.95, 0.51] |
| [bern-table4.png](bern-table4.png) | Table 4 / 8 | [PDF](https://arxiv.org/pdf/2611.02093v1) | [0.055, 0.075, 0.5, 0.228] |
| [replay-fig2.png](replay-fig2.png) | Figure 2 / 3 | [PDF](https://arxiv.org/pdf/2611.02746v2) | [0.075, 0.08, 0.925, 0.4] |
| [replay-fig3.png](replay-fig3.png) | Figure 3 / 9 | [PDF](https://arxiv.org/pdf/2611.02746v2) | [0.075, 0.09, 0.925, 0.5] |
| [replay-table2.png](replay-table2.png) | Table 2 / 16 | [PDF](https://arxiv.org/pdf/2611.02746v2) | [0.075, 0.085, 0.925, 0.565] |
| [decision-fig2.png](decision-fig2.png) | Figure 2 / 2 | [PDF](https://arxiv.org/pdf/2610.42128v1) | [0.055, 0.075, 0.95, 0.485] |
| [decision-fig3.png](decision-fig3.png) | Figure 3 / 7 | [PDF](https://arxiv.org/pdf/2610.42128v1) | [0.055, 0.425, 0.95, 0.76] |
| [decision-table2.png](decision-table2.png) | Table 2 / 10 | [PDF](https://arxiv.org/pdf/2610.42128v1) | [0.055, 0.07, 0.95, 0.3] |
| [decision-table3.png](decision-table3.png) | Table 3 / 10 | [PDF](https://arxiv.org/pdf/2610.42128v1) | [0.055, 0.505, 0.95, 0.742] |
| [spec-fig2.png](spec-fig2.png) | Figure 2 / 3 | [PDF](https://arxiv.org/pdf/2611.02846v1) | [0.1, 0.1, 0.9, 0.344] |
| [spec-fig5.png](spec-fig5.png) | Figure 5 / 5 | [PDF](https://arxiv.org/pdf/2611.02846v1) | [0.415, 0.465, 0.9, 0.745] |
| [spec-table1.png](spec-table1.png) | Table 1 / 6 | [PDF](https://arxiv.org/pdf/2611.02846v1) | [0.1, 0.1, 0.9, 0.465] |
| [spec-fig7.png](spec-fig7.png) | Figure 7 / 8 | [PDF](https://arxiv.org/pdf/2611.02846v1) | [0.51, 0.185, 0.9, 0.415] |
| [spec-fig8.png](spec-fig8.png) | Figure 8 / 8 | [PDF](https://arxiv.org/pdf/2611.02846v1) | [0.51, 0.425, 0.9, 0.86] |

全部图片均在report.md中标明条件、基线和读法；截图是论文原图，教学算例不画成伪实验。Bernoulli原文没有新网络架构图，使用其原始机制诊断图并如实说明。源PDF、截图、元数据证据及代码文件哈希见manifest。

## 固定代码与公开状态

### SH-OPD

Official repository linked by authors; selected core implementation and launcher/config files statically checked at pinned commit; no GPU training executed

- [README.md](https://github.com/chen123CtrlS/SH-OPD/blob/98467d23930e2197dd0b240af9e37cc7c5c3f505/README.md)：`485a14a3356bc2180eae1750f955365e1e71196abb1b6dea60c47174b7057b8d`
- [verl/verl/utils/random_horizon.py](https://github.com/chen123CtrlS/SH-OPD/blob/98467d23930e2197dd0b240af9e37cc7c5c3f505/verl/verl/utils/random_horizon.py)：`40d1e5565c90cde33e612dd17b89f383dc754e095d68b85643cb86ea2bbaef7d`
- [recipe/ratio_run.sh](https://github.com/chen123CtrlS/SH-OPD/blob/98467d23930e2197dd0b240af9e37cc7c5c3f505/recipe/ratio_run.sh)：`27125b82f55f4a5e142957225aa6d7005fc0243c077045c034bcdaf41f7e7b19`
- [recipe/opd_fulldata.sh](https://github.com/chen123CtrlS/SH-OPD/blob/98467d23930e2197dd0b240af9e37cc7c5c3f505/recipe/opd_fulldata.sh)：`1608023c252c5e374c1834d4afdc8364133e90a298a5a17a326baa5080c4ce73`
### Bernoulli OPD

Official repository linked by authors; selected core implementation and launcher/config files statically checked at pinned commit; no GPU training executed

- [README.md](https://github.com/duterscmy/Bernoulli-OPD/blob/21afbbbb93d46da9b1de5593e6fcdbe8baeee973/README.md)：`9582312efef3d4ea8c76be0fa9133a94a3ec41884b1613e60ebe3d3bd8e59c4d`
- [slime/backends/megatron_utils/opd.py](https://github.com/duterscmy/Bernoulli-OPD/blob/21afbbbb93d46da9b1de5593e6fcdbe8baeee973/slime/backends/megatron_utils/opd.py)：`fc05f252d892c112d4f4c675279d701dfdd37292924fd25db303d443d8091f6b`
- [configs/methods/bernoulli.env](https://github.com/duterscmy/Bernoulli-OPD/blob/21afbbbb93d46da9b1de5593e6fcdbe8baeee973/configs/methods/bernoulli.env)：`b4cf1d3f8cf499c3058c88c93ae0cfdf3888cbb32e991d8cdc78dc697401e633`
- [configs/settings/qwen3_4b.env](https://github.com/duterscmy/Bernoulli-OPD/blob/21afbbbb93d46da9b1de5593e6fcdbe8baeee973/configs/settings/qwen3_4b.env)：`0eef8df1f7eb92fae196638f4c08468a16ce03d338d2bb4a579dfec4294ed4a5`
### Continual MOPD Replay

Author-linked Zenodo API record verified: exact paper title, files contains only main.pdf, no training code artifact; LastOPD dependency is not this method release

### DecisionRank

No dedicated official method code verified in original PDF links, primary metadata and exact-title searches on 2026-11-21; not a claim of nonexistence

### SpecOPD

Official overview repository only: README Release Status explicitly excludes training code, evaluation scripts and weights; full tracked file inventory checked at pinned commit

- [README.md](https://github.com/QwenApp/SpecOPD/blob/7fe9f9f35b027691633ec424821df5985fe34576/README.md)：`a6978fe5dabdf01efbe0eee9116f8fc4951d80e70da3ac44298102d4ada9d8e1`

Replay关联Zenodo资源只验证到main.pdf，未冒称代码；DecisionRank未核实专门作者代码。SpecOPD官方仓库明确尚无训练/评测/权重发布。代码仅静态阅读，不代表实际训练验证。
