# Papers Daily

多方向论文精读：机制、公式逐符号解释与推导、手算直觉、实现、原图、实验条件、局限与复现。

## 更新安排

计划北京时间（Asia/Shanghai）每天09:00，由ChatGPT云端定时任务检索、写作、生成PDF并通过GitHub连接普通提交到main。任务启用与环境验证状态见 [AUTOMATION.md](AUTOMATION.md)。
每次执行都从仓库读取状态，不依赖个人电脑、SSH或本机路径。首个计划运行时间为2026-11-17 09:00。
近一个日历月内目标5篇新论文；按arXiv去版本ID、DOI、规范URL及规范标题跨日去重，不足如实报告，不用旧论文或博客凑数。

## 方向

- [VLM](VLM/README.md)

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

`方向/YYYY-MM-DD/` 内含report.md、同内容report.pdf、manifest.json与assets/；其他方向使用独立一级目录。
PDF嵌入Linux Noto中文字形；基础教程可早于近期窗口，但不计入每日5篇。原图注明论文图号、来源并逐图解释。
README与各方向索引及sources.json由scripts/update_index.py根据全部manifest重建。sources.json v2保存每个来源的全部去重别名。
每日发布和失败处理见 [AUTOMATION.md](AUTOMATION.md)。
