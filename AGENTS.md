# Papers Daily 仓库约定

- 添加、补录或调整日报前，先读 [提交规范](CONTRIBUTING.md) 及对应方向的运行约定。
- 根 README 每个方向只放最新一期，方向 README 保留最近 7 期。论文标题和历史日报进入本方向的月归档。
- 索引以各日 `manifest.json` 为准。运行 `python3 scripts/update_index.py` 生成索引，再用 `python3 scripts/update_index.py --check` 核对。旧日程中关于首页列出论文标题的描述，以此归档结构为准。
- 提交本次日报及相关生成文件。若远端出现并发更新，先保留该更新并基于最新清单重建索引。
