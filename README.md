# Papers Daily

多方向论文精读：机制、公式逐符号解释与推导、手算直觉、实现、原图、实验条件、局限与复现。

## 更新安排

计划北京时间（Asia/Shanghai）每天09:00，由ChatGPT云端定时任务检索、写作、生成PDF并通过GitHub连接普通提交到main。任务启用与环境验证状态见 [AUTOMATION.md](AUTOMATION.md)。
目标为每次从仓库读取状态、不依赖个人电脑。云端环境绑定尚待核验，托管日程暂停；迁移期间旧本地日程补位，仍依赖电脑开机。当前状态以AUTOMATION.md为准。
OPD & OPSD专项在北京时间每天09:15由既有本地日程执行，归档到OPSD/；上述09:00为VLM计划。
近一个日历月内目标5篇新论文；按arXiv去版本ID、DOI、规范URL及规范标题跨日去重，不足如实报告，不用旧论文或博客凑数。

## 方向

- [OPSD](OPSD/README.md)
- [VLM](VLM/README.md)

## 每日索引

| 日期 | 方向 / 子主题 | 数量 | 阅读 |
| --- | --- | --- | --- |
| 2026-11-17 | VLM / 小型VLM蒸馏、约束GRPO与能力保留 | 5 | [Markdown](VLM/2026-11-17/report.md) · [PDF](VLM/2026-11-17/report.pdf) |
| 2026-11-16 | VLM / vLLM推理系统 | 5 | [Markdown](VLM/2026-11-16/report.md) · [PDF](VLM/2026-11-16/report.pdf) |
| 2026-11-17 | OPSD / 监督选择、环境校准与异步加速 | 5 | [Markdown](OPSD/2026-11-17/report.md) · [PDF](OPSD/2026-11-17/report.pdf) |
| 2026-11-16 | OPSD / 特权信息、token级 credit 与应用扩展 | 5 | [Markdown](OPSD/2026-11-16/report.md) · [PDF](OPSD/2026-11-16/report.pdf) |

## 条目索引

### 2026-11-17 · 小型VLM蒸馏、约束GRPO与能力保留

1. [Teach with Blinks: Breaking Linguistic Momentum in On-Policy Distillation for Fine-Grained Multimodal Perception](https://arxiv.org/abs/2611.16402) · 2026-11-12 · [精读](VLM/2026-11-17/report.md)
2. [Safety as a Budget: Calibrating the Control Channel of Constrained GRPO for Vision-Language Models](https://arxiv.org/abs/2611.16628) · 2026-11-12 · [精读](VLM/2026-11-17/report.md)
3. [Erase the Injected Region: Multimodal Prompt-Injection Defense with On-Policy Self-Distillation](https://arxiv.org/abs/2610.41307) · 2026-10-30 · [精读](VLM/2026-11-17/report.md)
4. [From Foundation VLM to Versatile Driving VLM: A Triplet Incremental Training Approach](https://arxiv.org/abs/2611.16727) · 2026-11-12 · [精读](VLM/2026-11-17/report.md)
5. [From Tokens to Trajectories: Parameter-Efficient Continuous Adaptation of Unified Multimodal Models](https://arxiv.org/abs/2611.05451) · 2026-11-03 · [精读](VLM/2026-11-17/report.md)
### 2026-11-16 · vLLM推理系统

1. [DeepSeek-V4.1-Flash on Vera Rubin NVL72](https://vllm.ai/blog/2026-11-12-deepseek-v41-flash-vera-rubin) · 2026-11-12 · [精读](VLM/2026-11-16/report.md)
2. [Encoder Disaggregation in vLLM: Scaling Multimodal Serving with EPD](https://vllm.ai/blog/2026-11-11-encoder-disaggregation-epd) · 2026-11-11 · [精读](VLM/2026-11-16/report.md)
3. [Migrating vLLM to the LibTorch Stable ABI](https://vllm.ai/blog/2026-11-06-stable-abi-cuda) · 2026-11-06 · [精读](VLM/2026-11-16/report.md)
4. [Introducing Kinference](https://vllm.ai/blog/2026-11-04-introducing-kinference) · 2026-11-04 · [精读](VLM/2026-11-16/report.md)
5. [vLLM v0.33.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3) · 2026-11-02 · [精读](VLM/2026-11-16/report.md)
### 2026-11-17 · 监督选择、环境校准与异步加速

1. [ECOS: Environment-Calibrated On-Policy Self-Distillation for Language Agents](https://arxiv.org/abs/2611.12232) · 2026-11-09 · [精读](OPSD/2026-11-17/report.md)
2. [Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation](https://arxiv.org/abs/2611.08179) · 2026-11-05 · [精读](OPSD/2026-11-17/report.md)
3. [Shuffled OPSD: When Semantic Consistency Hurts On-Policy Self-Distillation?](https://arxiv.org/abs/2611.04958) · 2026-11-03 · [精读](OPSD/2026-11-17/report.md)
4. [Asynchronous On-Policy Self-Distillation](https://arxiv.org/abs/2610.25880) · 2026-10-19 · [精读](OPSD/2026-11-17/report.md)
5. [Not Every Rollout Needs Distillation: Correctness-Filtered On-Policy Self-Distillation](https://arxiv.org/abs/2610.25842) · 2026-10-19 · [精读](OPSD/2026-11-17/report.md)
### 2026-11-16 · 特权信息、token级 credit 与应用扩展

1. [Turning Multi-Agent Communication into Token-Level Credit with On Policy Self Distillation](https://arxiv.org/abs/2611.16206) · 2026-11-11 · [精读](OPSD/2026-11-16/report.md)
2. [Privileged Information Does Not Guarantee Privileged Guidance: Auditing On-Policy Self-Distillation for Reasoning Models](https://arxiv.org/abs/2611.16064) · 2026-11-11 · [精读](OPSD/2026-11-16/report.md)
3. [Prior-as-Skill: Remember Outside and Read Inside with On-Policy Self-Distillation for Self-Evolving VLM Agents](https://arxiv.org/abs/2611.15766) · 2026-11-11 · [精读](OPSD/2026-11-16/report.md)
4. [Bootstrapped On-Policy Distillation](https://arxiv.org/abs/2611.13680) · 2026-11-10 · [精读](OPSD/2026-11-16/report.md)
5. [PhysOPSD: Code as Privilege for Physically Plausible Video Generation](https://arxiv.org/abs/2611.12411) · 2026-11-09 · [精读](OPSD/2026-11-16/report.md)

## 仓库约定

`方向/YYYY-MM-DD/` 内含report.md、同内容report.pdf、manifest.json与assets/；其他方向使用独立一级目录。
PDF嵌入Linux Noto中文字形；基础教程可早于近期窗口，但不计入每日5篇。原图注明论文图号、来源并逐图解释。
README与各方向索引及sources.json由scripts/update_index.py根据全部manifest重建。sources.json v2保存每个来源的全部去重别名。
每日发布和失败处理见 [AUTOMATION.md](AUTOMATION.md)。
