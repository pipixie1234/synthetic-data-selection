# Verifier redundancy pilot

已完成实验见 [报告](outputs/redundancy_report.md) 和 [Excel](outputs/redundancy_results.xlsx)。

仅从现有 expanded_candidate_pool 抽取 160 条 synthetic，从冻结 RealReference 抽取 40 条，再复制这 40 条真实文本并改错标签构造 challenge。没有训练、重新生成新闻文本或读取最终 holdout。

`run_redundancy.py` 提供 prepare、verify、analyze 三阶段，`report_redundancy.py` 导出报告、Excel 与图。运行命令和协议限制均写入报告；`outputs/manifest.json` 记录输入 hash、seed 与 prompt 修订，原始输出、逐条判断和 cutoff ties 完整保留。

两模型最终使用同一份 source-blind v2 user prompt，采用各自 native chat wrapper。v1 的 Qwen-only 试跑因 confidence 语义错误而排除，保留在 `outputs/attempt1_confidence_semantics_failed`；最终指标不会混入 v1。

统计检查可用以下命令重跑：

```bash
/opt/anaconda/bin/python -m unittest discover -s SyntheticData/redundancy -p 'test_*.py' -v
```

输出属于单次开发性 pilot：低 agreement 需要先结合 challenge 可靠性、输出逻辑一致性与 confidence 并列情况解读，source 接受率差异不等于因果 self-bias。
