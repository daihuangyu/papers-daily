# Papers Daily

多方向论文与硬核技术资料精读。每日报告包含机制、公式、实现细节、原图、实验条件与分级延伸阅读。

## 更新安排

每天北京时间 09:00，由 Codex 当前聊天的定时任务检索、写作、生成 PDF 并提交。
README 由清单自动生成；同一来源 ID 或 URL 不允许重复。近期窗口内资料不足时报告实际数量，不用旧资料凑数。

## 方向

- [VLM](VLM/README.md)：小型视觉语言模型的训练与后训练，重点为OPD/蒸馏、RL、视觉推理和搜索agent；论文优先。首日vLLM工程资料作为历史内容保留。

## 每日索引

| 日期 | 方向 / 子主题 | 数量 | 阅读 |
| --- | --- | --- | --- |
| 2026-11-16 | VLM / vLLM推理系统 | 5 | [Markdown](VLM/2026-11-16/report.md) · [PDF](VLM/2026-11-16/report.pdf) |

## 条目索引

### 2026-11-16 · vLLM推理系统

1. [DeepSeek-V4.1-Flash on Vera Rubin NVL72](https://vllm.ai/blog/2026-11-12-deepseek-v41-flash-vera-rubin) · 2026-11-12 · [精读](VLM/2026-11-16/report.md)
2. [Encoder Disaggregation in vLLM: Scaling Multimodal Serving with EPD](https://vllm.ai/blog/2026-11-11-encoder-disaggregation-epd) · 2026-11-11 · [精读](VLM/2026-11-16/report.md)
3. [Migrating vLLM to the LibTorch Stable ABI](https://vllm.ai/blog/2026-11-06-stable-abi-cuda) · 2026-11-06 · [精读](VLM/2026-11-16/report.md)
4. [Introducing Kinference](https://vllm.ai/blog/2026-11-04-introducing-kinference) · 2026-11-04 · [精读](VLM/2026-11-16/report.md)
5. [vLLM v0.33.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3) · 2026-11-02 · [精读](VLM/2026-11-16/report.md)

## 仓库约定

`方向/YYYY-MM-DD/` 内含 report.md、report.pdf、manifest.json 与 assets/；scripts/ 管理 PDF、索引和发布。
基础教程可早于近期窗口，但不算入每日5篇。图片逐图注明出处，原图版权归原作者；教学图和推导明确标注。
每日发布流程见 [AUTOMATION.md](AUTOMATION.md)。
