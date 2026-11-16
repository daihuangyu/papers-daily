# 每日运行约定

- 仓库：https://github.com/daihuangyu/papers-daily
- 时间：北京时间每天09:00，由本聊天的 Codex 定时任务执行；不是 GitHub Actions 自行生成文章。运行依赖本机 Codex 可用及网络/SSH 授权正常。
- 内容：最近一个月的5篇小型VLM/MLLM训练与后训练论文，参考Vision-OPD和MMSearch-R1方向，覆盖on-policy蒸馏、RL/GRPO、偏好优化、视觉细粒度感知、视觉推理、搜索/工具agent。论文为主，博客仅辅助；2026-11-16的vLLM工程文档保留为首日例外。归档到VLM/YYYY-MM-DD/。
- 深度：每篇有问题、机制、公式逐符号解释、数值例子、实现或伪代码、原始实验条件、核心结果、局限和官方/作者入门链接。优先原文架构图和核心实验图，注明图号、出处、所比较基线，禁止生成虚假实验图。
- 去重：读 sources.json 和全部 manifest.json；按 DOI/arXiv去版本ID、官方规范URL与标题交叉核验。版本更新不默认算新资料。
- 不足5篇：保持近一个月和不重复的约束，写明找到的实际篇数与不足原因，不用旧资料或同一篇拆成几篇凑数。
- PDF：必须从最终Markdown全文生成，包含同样的图和图注，不用短版替代；渲染后检查。
- 提交：先确认工作区状态，保护其他方向改动；生成后运行 scripts/update_index.py，再运行 scripts/publish.sh。失败保留本地文件并报告错误，不能声称已推送。
- 远程：使用仓库本地配置的 origin，不使用浏览器登录凭据，不保存token。首次推送与每日推送均为main分支普通提交，不强制推送。

## 本地命令

```sh
python3 scripts/build_report.py VLM/2026-11-16/report.md
python3 scripts/update_index.py
sh scripts/publish.sh
```

PDF生成脚本当前使用macOS Songti字体；迁到Linux时需配置可嵌入的中文TrueType字体。
