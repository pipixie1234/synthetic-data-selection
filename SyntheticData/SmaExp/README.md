# AG News synthetic data experiment

Model files are expected under `/root/cloud/model`; this environment also falls back to `/root/workspace/cloud/model`. Set `SMAEXP_MODEL_ROOT` to override the location.

The local Qwen check currently fails because the indexed model is 8,045,591,552 bytes but only two partial shards and one small shard are present. Resume it with the domestic mirror after installing dependencies:

```powershell
py -m pip install -r requirements.txt
py download_models.py --skip-qwen
py verify_model.py /root/cloud/model/Qwen3-4B-Instruct-2507
```

`download_gptneo.ps1` uses `https://hf-mirror.com` and resumes the large file with HTTP range requests. The selected `gustavecortal/gpt-neo-2.7B-8bit` checkpoint is the closest practical GPT-Neo checkpoint to the requested roughly 4B size; its 8-bit weights are about 2.8 GB. On a CUDA setup with a compatible `bitsandbytes`, add `--four-bit` to generation when the checkpoint loader supports it. `download_models.py` remains available for repositories that work with the Hub snapshot API and for resuming Qwen metadata.

实验协议固定为：真实 AG News 训练集每类 50 条；Qwen 每类生成 100 条；GPT-Neo 每类生成 100 条；官方测试集每类固定抽取 100 条。`prepare` 会保存原始 split/index，并检查真实训练集与测试集没有完全相同文本；生成时也会过滤与真实训练/测试集完全重复的文本。

先建立真实数据划分：

```powershell
py synthetic_agnews.py prepare --protocol-dir outputs/protocol --real-train-per-label 50 --test-per-label 100
```

分别生成两种模型的数据：

```powershell
py synthetic_agnews.py generate --model qwen --protocol-dir outputs/protocol --num-per-label 100
py synthetic_agnews.py generate --model gpt --protocol-dir outputs/protocol --num-per-label 100
```

默认输出分别是 `outputs/protocol/synthetic_qwen.jsonl`、`synthetic_gpt.jsonl`；也可以用 `--model both` 生成合并文件 `synthetic_both.jsonl`。

Inspect intrinsic data quality before training:

```powershell
py synthetic_agnews.py evaluate --data outputs/ag_news_synthetic.jsonl
```

比较真实 50/class、Qwen、GPT-Neo、两模型合并，以及各自加入真实样本后的 BERT 分类结果：

```powershell
py synthetic_agnews.py compare --protocol-dir outputs/protocol
```

`compare` 使用同一份 400 条真实测试集，输出 `outputs/comparison_results.json`。BERT 分类头使用交叉熵损失（`AutoModelForSequenceClassification` 默认行为）。生成 JSONL 包含 `generator` 和原始来源字段；loader 会显式处理旧版 bitsandbytes blockwise 权重。

比较项包括：真实 50/class；Qwen 100/class；GPT-Neo 100/class；真实+Qwen；真实+GPT-Neo；如果存在合并文件，还包括真实+两模型。所有结果都在同一份 400 条测试集上评估。
