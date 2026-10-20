# E1/E2 Multi-Seed Validation

Run from `/root/workspace` with the cached AG News Arrow files and local model
weights available:

```bash
LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH \
python SyntheticData/exp/run_e1_e2_validation.py \
  --dev-size 2000 \
  --dev-seed 2026 \
  --selection-seeds 41 42 43 44 45 \
  --model-seeds 42 43 44 45 46 \
  --top-k 5 \
  --real-reference qc_train \
  --evaluate-legacy400 \
  --no-final-holdout
```

The command creates 25 E1 runs and 5 E2 runs under
`SyntheticData/exp/outputs/curation_validation/`. Existing completed runs are
reused when their `results.json` matches the requested model seed and train
count. `--evaluate-final-holdout` is opt-in and should only be used after the
method is frozen.

Primary outputs include `run_results.csv`, `e1_e2_validation_results.xlsx`,
`e1_e2_validation_analysis.txt`, `split_audit.json`, `dev2000.csv`,
`protocol_manifest.json`, and the fixed/random selected datasets.

## Two-GPU tmux run

Prepare the shared split, fixed E2 set, five E1 sets, and embedding cache once:

```bash
tmux new-session -d -s curation_prepare -c /root/workspace \
  'env LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python SyntheticData/exp/run_e1_e2_validation.py --prepare-only --dev-size 2000 --dev-seed 2026 --selection-seeds 41 42 43 44 45 --model-seeds 42 43 44 45 46 --top-k 5 --real-reference qc_train --evaluate-legacy400 --no-final-holdout > SyntheticData/exp/curation_prepare.log 2>&1'
```

After `curation_prepare.log` contains `Prepared validation artifacts`, launch
one shard per GPU:

```bash
tmux new-session -d -s curation_gpu0 -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python SyntheticData/exp/run_e1_e2_validation.py --dev-size 2000 --dev-seed 2026 --selection-seeds 41 42 43 44 45 --model-seeds 42 43 44 45 46 --top-k 5 --real-reference qc_train --evaluate-legacy400 --no-final-holdout --shard-index 0 --num-shards 2 > SyntheticData/exp/curation_gpu0.log 2>&1'

tmux new-session -d -s curation_gpu1 -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python SyntheticData/exp/run_e1_e2_validation.py --dev-size 2000 --dev-seed 2026 --selection-seeds 41 42 43 44 45 --model-seeds 42 43 44 45 46 --top-k 5 --real-reference qc_train --evaluate-legacy400 --no-final-holdout --shard-index 1 --num-shards 2 > SyntheticData/exp/curation_gpu1.log 2>&1'
```

Check progress:

```bash
tmux ls
tail -f /root/workspace/SyntheticData/exp/curation_gpu0.log
tail -f /root/workspace/SyntheticData/exp/curation_gpu1.log
```

Only after both shards report completion, merge and generate the final reports:

```bash
tmux new-session -d -s curation_merge -c /root/workspace \
  'env LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python SyntheticData/exp/run_e1_e2_validation.py --merge-shards --output-dir /root/workspace/SyntheticData/exp/outputs/curation_validation > SyntheticData/exp/curation_merge.log 2>&1'
```
