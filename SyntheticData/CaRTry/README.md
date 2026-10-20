# RA-CaR AG News experiment

This directory contains the Real-Anchored Clustering and Ranking adaptation
for the frozen AG News synthetic-data protocol. It is independent of the
previous DSIR directories.

Ranking is copied from E2's cached `top5_real_cosine_mean` score: the mean of
the five largest cosine similarities to same-class QC real-reference
embeddings. No DSIR score, quality weight, MMD allocation, or text-length
term is used.

For each source x class group, the fixed-budget selector takes the top
`40 - k_cluster` proximity examples and adds at most one highest-proximity
example from each semantic KMeans cluster. `k_cluster=round(sqrt(n/2))`, so
80 candidates produce six clusters. The final budget is always 40 per
source x class.

The environment does not currently have the `sentence-transformers` package
or a local MiniLM checkpoint. The implementation tries
`sentence-transformers/all-MiniLM-L6-v2` first and records the failure; in
this environment it uses the existing E2 RoBERTa embedding cache as the
semantic-clustering fallback. This is explicitly recorded in
`outputs/car_selector_validation/cache/car_cluster_embeddings.pt`.

Prepare selections and diagnostics:

```bash
cd /root/workspace
env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH \
python -u SyntheticData/CaRTry/run_car_validation.py \
  --prepare-only --evaluate-legacy400
```

Run the 25 ordinary BERT + CE jobs on the single 16GB GPU. The script writes
one result per job and reuses completed `results.json` files after an
interruption:

```bash
tmux new-session -d -s car_single -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH TOKENIZERS_PARALLELISM=false python -u SyntheticData/CaRTry/run_car_validation.py --evaluate-legacy400 --shard-index 0 --num-shards 1 > SyntheticData/CaRTry/car_single.log 2>&1'
```

Monitor:

```bash
tmux ls
tail -f /root/workspace/SyntheticData/CaRTry/car_single.log
find /root/workspace/SyntheticData/CaRTry/outputs/car_selector_validation/shards \
  -name results.json | wc -l
```

After the count reaches 25, merge with the existing Random, TopK, Original
DSIR, and LN-DSIR results without retraining those methods:

```bash
tmux new-session -d -s car_merge -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python -u SyntheticData/CaRTry/run_car_validation.py --evaluate-legacy400 --merge-shards > SyntheticData/CaRTry/car_merge.log 2>&1'
```

Final artifacts are under `outputs/car_selector_validation/`. The Excel
report is `car_selector_results.xlsx`; the text summary is
`car_selector_analysis.txt`.
