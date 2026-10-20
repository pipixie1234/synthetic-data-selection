# Cross-verifier downstream validation

按 `/root/workspace/newprompt.txt` 固定协议验证现有 4,000 条候选池的 selection utility 与 cross/self 差异。

主实验：Random320（5 selection seeds × 5 model seeds）、TopK320、SelfVerifier320、CrossVerifier320（后三项各5 model seeds），共40次。source-only的 self/cross辅助实验共20次，预算160；与主比较分开解释。

采用已冻结的 repaired real-anchored prompt、每类2条真实demonstrations、四标签完整序列likelihood。只在source×class内排序，不使用模型自报confidence，不平均模型尺度，不按Task B或标签匹配硬过滤。稳定log概率避免近1分数舍入导致假同分。

全部选集和diagnostics先冻结，再一次性进入固定3epoch的BERT CE训练与Dev2000最终评估；不读取Final Holdout、不重新生成synthetic、不根据Dev调整方法。

```bash
env LD_LIBRARY_PATH=/opt/anaconda/lib CUDA_VISIBLE_DEVICES=0 /opt/anaconda/bin/python SyntheticData/cross_verifier/run_pipeline.py
env LD_LIBRARY_PATH=/opt/anaconda/lib /opt/anaconda/bin/python -m unittest discover -s SyntheticData/cross_verifier -p 'test_*.py' -v
```

pipeline支持hash校验与断点复用；每阶段日志、4000×2模型likelihood、所有selected IDs、60次run-level指标和Dev预测均留存在outputs。

完成结果见 [报告](outputs/cross_verifier_report.md)、[Excel](outputs/cross_verifier_summary.xlsx)、[逐次训练结果](outputs/cross_verifier_run_results.csv)、[图](outputs/cross_verifier_summary.png)。`pipeline_status.json` 表示执行状态，报告只在全部运行完成后生成。
