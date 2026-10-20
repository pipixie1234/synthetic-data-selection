# 同一 200 条数据上的 verifier 修复实验

Task A 只预测 AG News 标签，比较 zero-shot 与每类 2 条真实示例的 real-anchored。Task B 独立判断 news_like/coherent/plausible/template_like，由代码计算 LabelPass 与 OverallPass。

采用本地 Qwen3-4B 与 Llama-3.2-3B；相同评测文本、相同 prompts、示例与评测双重排除。排序使用四个完整标签候选的 log probability，不再使用模型自报 confidence。Task B 每模型只运行一次，供两个标签配置共用。

完成结果见 [实验报告](outputs/verifier_repair_report.md) 与 [Excel](outputs/verifier_repair_results.xlsx)。报告包含完整运行命令和指标适用范围。

输出位于单独目录 `outputs/`，不会改写上一轮 `SyntheticData/redundancy/outputs/`。
