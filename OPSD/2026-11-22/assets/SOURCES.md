# 原图、原文与代码来源

核查日期：2026-11-22。指定版本PDF以2×渲染后裁切；坐标为整页左/上/右/下0—1比例。未重绘或更改实验数据，版权归原作者/发布者。原始PDF与日期页面快照留存于研究工作目录，哈希见manifest。

| 文件 | 原图表 / PDF页 | 来源版本 | 裁切坐标 |
| --- | --- | --- | --- |
| [rds-fig2.png](rds-fig2.png) | Figure 2 / 3 | [PDF](https://arxiv.org/pdf/2611.12012v1) | [0.085, 0.08, 0.89, 0.315] |
| [rds-table1.png](rds-table1.png) | Table 1 / 7 | [PDF](https://arxiv.org/pdf/2611.12012v1) | [0.085, 0.08, 0.89, 0.383] |
| [rds-fig4.png](rds-fig4.png) | Figure 4 / 8 | [PDF](https://arxiv.org/pdf/2611.12012v1) | [0.5, 0.08, 0.89, 0.415] |
| [rds-fig6.png](rds-fig6.png) | Figure 6 / 9 | [PDF](https://arxiv.org/pdf/2611.12012v1) | [0.5, 0.08, 0.89, 0.222] |
| [esc-fig1.png](esc-fig1.png) | Figure 1 / 2 | [PDF](https://arxiv.org/pdf/2610.39020v1) | [0.115, 0.085, 0.89, 0.363] |
| [esc-fig3.png](esc-fig3.png) | Figure 3 / 7 | [PDF](https://arxiv.org/pdf/2610.39020v1) | [0.115, 0.405, 0.49, 0.72] |
| [esc-table2.png](esc-table2.png) | Table 2 / 8 | [PDF](https://arxiv.org/pdf/2610.39020v1) | [0.115, 0.085, 0.89, 0.242] |
| [esc-fig6.png](esc-fig6.png) | Figure 6 / 13 | [PDF](https://arxiv.org/pdf/2610.39020v1) | [0.115, 0.265, 0.89, 0.745] |
| [p2-fig2.png](p2-fig2.png) | Figure 2 / 4 | [PDF](https://arxiv.org/pdf/2611.03953v2) | [0.085, 0.085, 0.89, 0.37] |
| [p2-table1.png](p2-table1.png) | Table 1 / 7 | [PDF](https://arxiv.org/pdf/2611.03953v2) | [0.085, 0.08, 0.89, 0.465] |
| [p2-table4.png](p2-table4.png) | Table 4 / 8 | [PDF](https://arxiv.org/pdf/2611.03953v2) | [0.5, 0.08, 0.89, 0.31] |
| [p2-table5.png](p2-table5.png) | Table 5 / 9 | [PDF](https://arxiv.org/pdf/2611.03953v2) | [0.5, 0.08, 0.89, 0.27] |
| [rpsd-fig2.png](rpsd-fig2.png) | Figure 2 / 4 | [PDF](https://arxiv.org/pdf/2610.37281v1) | [0.17, 0.105, 0.835, 0.395] |
| [rpsd-table1.png](rpsd-table1.png) | Table 1 / 6 | [PDF](https://arxiv.org/pdf/2610.37281v1) | [0.17, 0.098, 0.835, 0.335] |
| [rpsd-table3.png](rpsd-table3.png) | Table 3 / 7 | [PDF](https://arxiv.org/pdf/2610.37281v1) | [0.17, 0.25, 0.835, 0.35] |
| [rpsd-tables6-7.png](rpsd-tables6-7.png) | Tables 6 and 7 / 23 | [PDF](https://arxiv.org/pdf/2610.37281v1) | [0.17, 0.098, 0.835, 0.39] |
| [sa-fig2.png](sa-fig2.png) | Figure 2 / 4 | [PDF](https://arxiv.org/pdf/2611.01568v1) | [0.083, 0.105, 0.92, 0.52] |
| [sa-table2.png](sa-table2.png) | Table 2 / 7 | [PDF](https://arxiv.org/pdf/2611.01568v1) | [0.083, 0.1, 0.92, 0.49] |
| [sa-fig3.png](sa-fig3.png) | Figure 3 / 8 | [PDF](https://arxiv.org/pdf/2611.01568v1) | [0.083, 0.1, 0.495, 0.38] |
| [sa-fig5.png](sa-fig5.png) | Figure 5 / 9 | [PDF](https://arxiv.org/pdf/2611.01568v1) | [0.083, 0.1, 0.495, 0.345] |

全部图片在report.md分别标明实验条件、基线、图号、页码及读法。20张均为原论文图表，包括架构、主结果、成本与消融；正文手算不是实测曲线。

## 公开代码与固定版本

### RDS-MOPD

No dedicated official method code verified in original PDF links, primary metadata and exact-title/method searches on 2026-11-22; not a claim of nonexistence

### ESC

Official author-linked repository; selected implementation and launchers statically inspected at pinned commit; no GPU training executed

- [verl_opd/README.md](https://github.com/maybenotime/opd-esc/blob/b0c1a6126d87e7c2197f4860ef9cbc2390176d7e/verl_opd/README.md)：`7781acd85cff4cb3f6bf7293fab25fc31b17313d245d96e674c94e6f87b9e0b6`
- [verl_opd/plugins/k1_probe/k1_probe.py](https://github.com/maybenotime/opd-esc/blob/b0c1a6126d87e7c2197f4860ef9cbc2390176d7e/verl_opd/plugins/k1_probe/k1_probe.py)：`b21246367691c0bf1545596c305b0260ee8ef728475af6d03b623d33d1a7481b`
- [verl_opd/plugins/k1_probe/tp_stage/tp_core.py](https://github.com/maybenotime/opd-esc/blob/b0c1a6126d87e7c2197f4860ef9cbc2390176d7e/verl_opd/plugins/k1_probe/tp_stage/tp_core.py)：`6cb657775bd9a5b99768becc68156f1b68b68316c20aabc594f867007e1bb420`
- [verl_opd/plugins/k1_probe/tp_stage/k1_transplant_loss.py](https://github.com/maybenotime/opd-esc/blob/b0c1a6126d87e7c2197f4860ef9cbc2390176d7e/verl_opd/plugins/k1_probe/tp_stage/k1_transplant_loss.py)：`3adb005354b21fbe2c3fca0e727171f53988c0b5e0e6a03117db1993c92e53ee`
- [verl_opd/scripts/chain19_tp.sh](https://github.com/maybenotime/opd-esc/blob/b0c1a6126d87e7c2197f4860ef9cbc2390176d7e/verl_opd/scripts/chain19_tp.sh)：`f2d4c4c8cf9aedc718ea44925c5eaee072ec955792d2d40d2c90665024e08389`
- [verl_opd/scripts/run_opd_math.sh](https://github.com/maybenotime/opd-esc/blob/b0c1a6126d87e7c2197f4860ef9cbc2390176d7e/verl_opd/scripts/run_opd_math.sh)：`21b08ff245936d5ba8edeeb092a35d3e6b7c9c08a1ccf9810b14bb462863f936`
### P²-MOPD

Official overview repository only: complete tracked tree has .gitignore, README.md and two PNG files; no training code, evaluation scripts or weights at pinned commit

- [README.md](https://github.com/SinuoWang/P2-MOPD/blob/68b19c5549f4fe7ed9cf43259fd8312b3476531a/README.md)：`3a13aec82015d3ad9596f9d18ac3d098e42ac29472ee487a71d8dc24df697151`
### RPSD

No dedicated official method code verified in original PDF links, primary metadata and exact-title/method searches on 2026-11-22; not a claim of nonexistence

### SA-OPSD

No dedicated official method code verified in original PDF links, primary metadata and exact-title/method searches on 2026-11-22; not a claim of nonexistence


P²-MOPD不能因README自称official implementation就视作公开训练代码；RDS、RPSD和本篇SA-OPSD未核实专门实现，不借用依赖或同缩写论文代码。ESC代码核查仅静态阅读，未执行GPU训练。独立作者博客未核实时明确标注，辅助教程不计入五篇。
