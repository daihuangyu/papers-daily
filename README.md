# Papers Daily

多方向论文精读：机制、公式逐符号解释与推导、手算直觉、实现、原图、实验条件、局限与复现。

## 更新安排

计划北京时间（Asia/Shanghai）每天09:00，由ChatGPT云端定时任务检索、写作、生成PDF并通过GitHub连接普通提交到main。任务启用与环境验证状态见 [AUTOMATION.md](AUTOMATION.md)。
目标为每次从仓库读取状态、不依赖个人电脑。云端环境绑定尚待核验，托管日程暂停；迁移期间旧本地日程补位，仍依赖电脑开机。当前状态以AUTOMATION.md为准。
OPD & OPSD专项在北京时间每天09:15由既有本地日程执行，归档到OPSD/；上述09:00为VLM计划。
VLM与OPSD按最近一个日历月筛选；CCF_Papers按当年录用年度筛选，计划北京时间每天10:00，不限制预印本首次公开月份。
每方向目标5篇；按arXiv去版本ID、DOI、规范URL、规范标题及改题别名跨日期与方向去重，不足如实报告，不用博客凑数。

## 方向

- [CCF_Papers](CCF_Papers/README.md)
- [OPSD](OPSD/README.md)
- [VLM](VLM/README.md)

## 每日索引

| 日期 | 方向 / 子主题 | 数量 | 阅读 |
| --- | --- | --- | --- |
| 2026-11-19 | VLM / 0.8B/2B证据蒸馏、视频依赖审计、多教师后续RL与循环定位 | 5 | [Markdown](VLM/2026-11-19/report.md) · [PDF](VLM/2026-11-19/report.pdf) |
| 2026-11-18 | VLM / 视觉证据对齐、视频信用分配、2B数据选择与持续后训练遗忘 | 5 | [Markdown](VLM/2026-11-18/report.md) · [PDF](VLM/2026-11-18/report.pdf) |
| 2026-11-17 | VLM / 小型VLM蒸馏、约束GRPO与能力保留 | 5 | [Markdown](VLM/2026-11-17/report.md) · [PDF](VLM/2026-11-17/report.pdf) |
| 2026-11-16 | VLM / vLLM推理系统 | 5 | [Markdown](VLM/2026-11-16/report.md) · [PDF](VLM/2026-11-16/report.pdf) |
| 2026-11-18 | OPSD / 教师进化、逐token选择与参考视图边缘化 | 5 | [Markdown](OPSD/2026-11-18/report.md) · [PDF](OPSD/2026-11-18/report.pdf) |
| 2026-11-17 | OPSD / 监督选择、环境校准与异步加速 | 5 | [Markdown](OPSD/2026-11-17/report.md) · [PDF](OPSD/2026-11-17/report.pdf) |
| 2026-11-16 | OPSD / 特权信息、token级 credit 与应用扩展 | 5 | [Markdown](OPSD/2026-11-16/report.md) · [PDF](OPSD/2026-11-16/report.pdf) |
| 2026-11-18 | CCF_Papers / 采样式SFT、一步生成、知识暴露、量化推理与后训练预算 | 5 | [Markdown](CCF_Papers/2026-11-18/report.md) · [PDF](CCF_Papers/2026-11-18/report.pdf) |

## 条目索引

### 2026-11-19 · 0.8B/2B证据蒸馏、视频依赖审计、多教师后续RL与循环定位

1. [Tiny Evidence Region Localization with Selective Contrastive On-Policy Self-Distillation](https://arxiv.org/abs/2611.10734) · 2026-11-07 · [精读](VLM/2026-11-19/report.md)
2. [Watching Itself Watch: Self-Auditing Visual Reliance for Video Reasoning](https://arxiv.org/abs/2611.11949) · 2026-11-09 · [精读](VLM/2026-11-19/report.md)
3. [M3OPD-Reasoner: An Open Recipe for Post-Training Multimodal Reasoning with Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2611.23034) · 2026-11-17 · [精读](VLM/2026-11-19/report.md)
4. [Generalized CycleGRPO for Fine-Grained Multimodal Understanding and Grounding](https://arxiv.org/abs/2610.29552) · 2026-10-21 · [精读](VLM/2026-11-19/report.md)
5. [PROVE: Probabilistic Visual Grounding via Patch-Level Evidence for Vision-Language Models](https://arxiv.org/abs/2610.32440) · 2026-10-23 · [精读](VLM/2026-11-19/report.md)
### 2026-11-18 · 视觉证据对齐、视频信用分配、2B数据选择与持续后训练遗忘

1. [Reframe: Observation-Aligned Multi-Teacher On-Policy Distillation for Agentic Visual Reasoning](https://arxiv.org/abs/2611.12886) · 2026-11-09 · [精读](VLM/2026-11-18/report.md)
2. [Uncovering Shortcut Amplification in Video RLVR through the Lens of Perceptual Uncertainty Collapse](https://arxiv.org/abs/2610.37211) · 2026-10-27 · [精读](VLM/2026-11-18/report.md)
3. [RECAP: Looking Once Is Not Enough for Vision-Language Reasoning](https://arxiv.org/abs/2610.30834) · 2026-10-22 · [精读](VLM/2026-11-18/report.md)
4. [DISCOVER: Online Variance-Guided Data Discovery for Budgeted Multimodal GRPO](https://arxiv.org/abs/2611.11145) · 2026-11-08 · [精读](VLM/2026-11-18/report.md)
5. [Continual Adaptation Across Modalities: Learning and Forgetting in Unified Multimodal Models](https://arxiv.org/abs/2610.38436) · 2026-10-28 · [精读](VLM/2026-11-18/report.md)
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
### 2026-11-18 · 教师进化、逐token选择与参考视图边缘化

1. [On-Policy Self-Distillation with Teacher Evolution](https://arxiv.org/abs/2610.38637) · 2026-10-28 · [精读](OPSD/2026-11-18/report.md)
2. [Every Token Needs Its Own Teacher: Multi-teacher Adaptive On-policy Self-distillation](https://arxiv.org/abs/2610.40574) · 2026-10-29 · [精读](OPSD/2026-11-18/report.md)
3. [One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation](https://arxiv.org/abs/2610.30880) · 2026-10-22 · [精读](OPSD/2026-11-18/report.md)
4. [TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation](https://arxiv.org/abs/2610.29167) · 2026-10-21 · [精读](OPSD/2026-11-18/report.md)
5. [Steering the Teacher: Improving On-Policy Self-Distillation for Thinking Models](https://arxiv.org/abs/2610.34005) · 2026-10-25 · [精读](OPSD/2026-11-18/report.md)
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
### 2026-11-18 · 采样式SFT、一步生成、知识暴露、量化推理与后训练预算

1. [Finetuning with Sampling: SFT Learns Better Than You Think](https://arxiv.org/abs/2610.02140) · 2026-10-01 · [精读](CCF_Papers/2026-11-18/report.md)
2. [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) · 2026-02-04 · [精读](CCF_Papers/2026-11-18/report.md)
3. [LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure](https://arxiv.org/abs/2608.13545) · 2026-08-13 · [精读](CCF_Papers/2026-11-18/report.md)
4. [Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206) · 2026-05-29 · [精读](CCF_Papers/2026-11-18/report.md)
5. [Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs](https://arxiv.org/abs/2609.01573) · 2026-09-01 · [精读](CCF_Papers/2026-11-18/report.md)

## 仓库约定

`方向/YYYY-MM-DD/` 内含report.md、同内容report.pdf、manifest.json与assets/；其他方向使用独立一级目录。
PDF嵌入中文字体，实际执行环境与字体记录在当日assets/QA.md；Linux云端默认使用Noto。基础教程可早于近期窗口，但不计入每日5篇。原图注明论文图号、来源并逐图解释。
README与各方向索引及sources.json由scripts/update_index.py根据全部manifest重建。sources.json v2保存每个来源的全部去重别名。
每日发布和失败处理见 [AUTOMATION.md](AUTOMATION.md)。
