# 原图、原文与代码来源

核查日期：2026-11-20。原始PDF以2×渲染后裁图，坐标为相对整页的左/上/右/下0—1比例；没有重绘、拼接或改动测量数据。图表版权归原作者/发布者。

| 文件 | 原图表 / PDF页 | 指定版本 | 裁切坐标 |
| --- | --- | --- | --- |
| [stone-fig3.png](stone-fig3.png) | Figure 3 / 4 | [PDF](https://arxiv.org/pdf/2611.18916v1) | [0.11, 0.085, 0.9, 0.425] |
| [stone-table1.png](stone-table1.png) | Table 1 / 5 | [PDF](https://arxiv.org/pdf/2611.18916v1) | [0.115, 0.08, 0.9, 0.5] |
| [stone-fig5.png](stone-fig5.png) | Figure 5 / 6 | [PDF](https://arxiv.org/pdf/2611.18916v1) | [0.51, 0.075, 0.91, 0.395] |
| [stone-table4.png](stone-table4.png) | Table 4 / 12 | [PDF](https://arxiv.org/pdf/2611.18916v1) | [0.11, 0.485, 0.515, 0.835] |
| [brief-fig1.png](brief-fig1.png) | Figure 1 / 2 | [PDF](https://arxiv.org/pdf/2611.19273v2) | [0.17, 0.075, 0.84, 0.425] |
| [brief-fig3.png](brief-fig3.png) | Figure 3 / 6 | [PDF](https://arxiv.org/pdf/2611.19273v2) | [0.17, 0.11, 0.84, 0.326] |
| [brief-table2.png](brief-table2.png) | Table 2 / 7 | [PDF](https://arxiv.org/pdf/2611.19273v2) | [0.165, 0.095, 0.84, 0.49] |
| [brief-tables4-5.png](brief-tables4-5.png) | Tables 4 and 5 / 9 | [PDF](https://arxiv.org/pdf/2611.19273v2) | [0.165, 0.08, 0.84, 0.485] |
| [copd-fig2.png](copd-fig2.png) | Figure 2 / 4 | [PDF](https://arxiv.org/pdf/2611.23673v1) | [0.11, 0.13, 0.9, 0.47] |
| [copd-table1.png](copd-table1.png) | Table 1 / 7 | [PDF](https://arxiv.org/pdf/2611.23673v1) | [0.11, 0.205, 0.9, 0.53] |
| [copd-fig5.png](copd-fig5.png) | Figure 5 / 9 | [PDF](https://arxiv.org/pdf/2611.23673v1) | [0.11, 0.07, 0.9, 0.365] |
| [copd-fig7.png](copd-fig7.png) | Figure 7 / 10 | [PDF](https://arxiv.org/pdf/2611.23673v1) | [0.11, 0.165, 0.9, 0.375] |
| [sd-fig1.png](sd-fig1.png) | Figure 1 / 2 | [PDF](https://arxiv.org/pdf/2611.07722v2) | [0.055, 0.075, 0.95, 0.375] |
| [sd-fig2.png](sd-fig2.png) | Figure 2 / 3 | [PDF](https://arxiv.org/pdf/2611.07722v2) | [0.055, 0.075, 0.95, 0.315] |
| [sd-table1.png](sd-table1.png) | Table 1 / 6 | [PDF](https://arxiv.org/pdf/2611.07722v2) | [0.055, 0.075, 0.95, 0.65] |
| [sd-table2.png](sd-table2.png) | Table 2 / 7 | [PDF](https://arxiv.org/pdf/2611.07722v2) | [0.055, 0.075, 0.95, 0.2] |
| [order-fig2.png](order-fig2.png) | Figure 2 / 4 | [PDF](https://arxiv.org/pdf/2611.02112v1) | [0.165, 0.1, 0.84, 0.34] |
| [order-table3.png](order-table3.png) | Table 3 / 7 | [PDF](https://arxiv.org/pdf/2611.02112v1) | [0.17, 0.11, 0.84, 0.497] |
| [order-table4.png](order-table4.png) | Table 4 / 8 | [PDF](https://arxiv.org/pdf/2611.02112v1) | [0.17, 0.305, 0.55, 0.52] |
| [order-fig5.png](order-fig5.png) | Figure 5 / 9 | [PDF](https://arxiv.org/pdf/2611.02112v1) | [0.17, 0.1, 0.84, 0.305] |

所有图表在report.md内有中文条件、基线及读图解释；图注中的页码为PDF文件页。PNG与源PDF SHA-256见manifest。源PDF在研究工作目录保留，不重复上传整篇论文。

## 公开代码核查

### B-OPSD

固定提交：8df0eae3424c9d03e82fa75346388e016a7dec05。只读源码，未运行训练。

- [README.md](https://github.com/ZevineXu/B-OPSD/blob/8df0eae3424c9d03e82fa75346388e016a7dec05/README.md)：`131e7edb8028c5ba05d17015323cc5d1dada2736cdd64e7a9ed61fbed2d75cb4`
- [b_opsd_train.py](https://github.com/ZevineXu/B-OPSD/blob/8df0eae3424c9d03e82fa75346388e016a7dec05/b_opsd_train.py)：`a6bd8b2353b3446f83eefd8c62a02eb9bd8b764e72de750ccee376078570a3a7`
- [b_opsd_trainer.py](https://github.com/ZevineXu/B-OPSD/blob/8df0eae3424c9d03e82fa75346388e016a7dec05/b_opsd_trainer.py)：`1af92e26c75049b71ba6c3d9adf5e8dfce8b352b432f3e32510259c5f74865ca`
- [opsd_train.py](https://github.com/ZevineXu/B-OPSD/blob/8df0eae3424c9d03e82fa75346388e016a7dec05/opsd_train.py)：`fd0b97fa0cb887ebe4181653b9d001776bc43caf3b2057245b5035eb9095446e`
- [data_collator.py](https://github.com/ZevineXu/B-OPSD/blob/8df0eae3424c9d03e82fa75346388e016a7dec05/data_collator.py)：`5b96f3b3ae2f04e2b9dbf03509d3cd723c1b637d9362ab4697135d9f885b8381`
- [scripts/run_b_opsd_1b.sh](https://github.com/ZevineXu/B-OPSD/blob/8df0eae3424c9d03e82fa75346388e016a7dec05/scripts/run_b_opsd_1b.sh)：`e68b952bd00bdb72910dcfce7123f7db3b73a3206cfe166e1d2d3f40b41d92e4`
- [scripts/run_opsd_1b.sh](https://github.com/ZevineXu/B-OPSD/blob/8df0eae3424c9d03e82fa75346388e016a7dec05/scripts/run_opsd_1b.sh)：`cb9c99bf20d1b35a8a422ee2d92c74b73467da6bccfbd7f053cd56c8cc2c3287`
### SD-OPD

固定提交：ae19768776759d3788c4c5a65d5562f2b0929eba。只读源码，未运行训练。

- [README.md](https://github.com/aim-uofa/sd-opd/blob/ae19768776759d3788c4c5a65d5562f2b0929eba/README.md)：`23d96ceeacbd68ef4c6ae7eea2c1d489cd18cf9c74490fdc96fd7a9ff8dc9b64`
- [train/sd_opd/run_sd_opd.sh](https://github.com/aim-uofa/sd-opd/blob/ae19768776759d3788c4c5a65d5562f2b0929eba/train/sd_opd/run_sd_opd.sh)：`fe95d00be8353bbe67dd5d7c8a17008d81e111df4b4bf2de2d734fdf61066d28`
- [train/run_opd.sh](https://github.com/aim-uofa/sd-opd/blob/ae19768776759d3788c4c5a65d5562f2b0929eba/train/run_opd.sh)：`8bad8c610d9f72480d78bf7d89b4e073a463675cc9d806aa0d3558f6d18eae61`
- [train/config.sh](https://github.com/aim-uofa/sd-opd/blob/ae19768776759d3788c4c5a65d5562f2b0929eba/train/config.sh)：`8a173a94a97cc18257f8e70e6cb9b9542d522182c168abc84c96c5dfa94d205b`
- [slime/slime/utils/student_base_delta.py](https://github.com/aim-uofa/sd-opd/blob/ae19768776759d3788c4c5a65d5562f2b0929eba/slime/slime/utils/student_base_delta.py)：`2cfe4ee8bc50ba36102a1d8bc618a2f7dc0bbe5ce7a5514fb06eec196ca404d5`
- [slime/slime/utils/distillation_scoring.py](https://github.com/aim-uofa/sd-opd/blob/ae19768776759d3788c4c5a65d5562f2b0929eba/slime/slime/utils/distillation_scoring.py)：`87d7a657dc9fa71525b16c8dfb172a9f6e8b70f25f5ede7c71c0e7c66ab7ec06`
- [slime/slime/backends/megatron_utils/student_base_delta.py](https://github.com/aim-uofa/sd-opd/blob/ae19768776759d3788c4c5a65d5562f2b0929eba/slime/slime/backends/megatron_utils/student_base_delta.py)：`637a55c42ce667e6debdd2e95961e00e233bcd228fee538195a575f4fae4de91`

B-OPSD长度/retry、归一化与本地数据路径差异，以及SD-OPD入口direct/current、topK近似和不同clipping条件均已在正文说明。COPD只有作者项目讲解页被核验，STONE/O3PD未找到专门作者方法代码；框架和第三方实现不冒称方法发布。
