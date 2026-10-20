# ARA-TopK

This is the final selector experiment. It uses the frozen E2 Top-5
same-class Real-Proximity score, clusters only the MiniLM QC real-reference
embeddings, maps Qwen/Llama candidates to the nearest real cluster, and
applies the specified JSD-derived adaptive quota. It never uses DSIR, DIMP,
MMD, Dev scores, or final holdout data.

Prepare both cluster seeds and inspect `ara_selection_diagnostics.csv`:

```bash
cd /root/workspace
python -u SyntheticData/APT/run_ara_topk.py --prepare-only --evaluate-legacy400
```

Run the ten ARA-TopK BERT+CE jobs on one GPU:

```bash
tmux new-session -d -s ara_topk -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python -u SyntheticData/APT/run_ara_topk.py --bert-model /root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 --evaluate-legacy400 > SyntheticData/APT/ara_topk.log 2>&1'
```

Monitor:

```bash
tmux attach -t ara_topk
```

The compact final workbook is `outputs/ara_topk_validation/ARA_MAIN_TABLE.xlsx`.
The main raw artifacts are `ara_selection_diagnostics.csv`,
`ara_run_results.csv`, `ara_paired_deltas.csv`, and `ara_cluster_variance.csv`.
