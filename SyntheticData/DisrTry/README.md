# DSIR selector experiments

The original class-conditional DSIR experiment remains under
`outputs/dsir_selector_validation/` and is not overwritten. The new
length-normalized experiment is implemented in `run_ln_dsir_validation.py`
and writes to `outputs/ln_dsir_validation/`.

Its selector score is exactly `(log_p_target - log_q_raw) / max(num_ngrams, 1)`.
The target is QC train160 only. Selection remains independent for each
source x class group, with 40 Qwen and 40 Llama examples per class. BERT
training uses ordinary unweighted cross entropy; selector weights are used
only before training.

The original experiment reuses the implementation in
`/root/workspace/dsir-main`:

- `data_selection.hashed_ngram_dsir.get_ngram_counts`
- `data_selection.hashed_ngram_dsir.hash_buckets`
- `wordpunct` tokenization
- 10,000 hashed buckets
- unigram + bigram features
- target smoothing coefficient `0.0`
- `log(prob + 1e-8)` importance scores
- Gumbel top-k weighted sampling without replacement

The method is named **Class-Conditional DSIR (CC-DSIR)**. It runs separately
inside each generator x class pool and always selects 40 Qwen plus 40 Llama
examples per class. Dev2000 is evaluation-only; final holdout is never
evaluated by default.

Prepare original E5 selection datasets and diagnostics:

```bash
LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH \
python SyntheticData/DisrTry/run_cc_dsir_validation.py \
  --prepare-only --evaluate-legacy400
```

The new LN-DSIR diagnostics-only command is:

```bash
env LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH \
python -u SyntheticData/DisrTry/run_ln_dsir_validation.py \
  --selector dsir_length_norm --prepare-only
```

It stops before training if severe-collapse criteria are triggered. Use
`--force-train` only for an explicit exploratory override. Once diagnostics
pass, run the 25 LN-DSIR BERT-CE jobs on two GPUs:

```bash
tmux new-session -d -s ln_dsi_gpu0 -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH TOKENIZERS_PARALLELISM=false python -u SyntheticData/DisrTry/run_ln_dsir_validation.py --selector dsir_length_norm --evaluate-legacy400 --shard-index 0 --num-shards 2 > SyntheticData/DisrTry/ln_dsi_gpu0.log 2>&1'

tmux new-session -d -s ln_dsi_gpu1 -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH TOKENIZERS_PARALLELISM=false python -u SyntheticData/DisrTry/run_ln_dsir_validation.py --selector dsir_length_norm --evaluate-legacy400 --shard-index 1 --num-shards 2 > SyntheticData/DisrTry/ln_dsi_gpu1.log 2>&1'
```

After both shards finish, merge and generate the four-way report:

```bash
tmux new-session -d -s ln_dsi_merge -c /root/workspace \
  'env LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python -u SyntheticData/DisrTry/run_ln_dsir_validation.py --selector dsir_length_norm --evaluate-legacy400 --merge-shards > SyntheticData/DisrTry/ln_dsi_merge.log 2>&1'
```

For reference, the old E5 command is still available below.

Then run the 25 E5 BERT-CE jobs on two GPUs:

```bash
tmux new-session -d -s dsir_gpu0 -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python SyntheticData/DisrTry/run_cc_dsir_validation.py --evaluate-legacy400 --shard-index 0 --num-shards 2 > SyntheticData/DisrTry/dsir_gpu0.log 2>&1'

tmux new-session -d -s dsir_gpu1 -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python SyntheticData/DisrTry/run_cc_dsir_validation.py --evaluate-legacy400 --shard-index 1 --num-shards 2 > SyntheticData/DisrTry/dsir_gpu1.log 2>&1'
```

After both shards finish, merge E5 with the existing E1/E2 baseline:

```bash
tmux new-session -d -s dsir_merge -c /root/workspace \
  'env LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python SyntheticData/DisrTry/run_cc_dsir_validation.py --merge-shards > SyntheticData/DisrTry/dsir_merge.log 2>&1'
```

Main outputs are under
`SyntheticData/DisrTry/outputs/dsir_selector_validation/`, including
`dsir_candidate_weights.csv`, `dsir_selection_frequency.csv`, the five
`e5_dsir_sel*.csv` datasets, `dsir_protocol_manifest.json`,
`dsir_selector_results.xlsx`, and `dsir_selector_analysis.txt`.

LN outputs are under `SyntheticData/DisrTry/outputs/ln_dsir_validation/`,
including `ln_dsir_weight_diagnostics.csv`, `ess_comparison.csv`,
`length_sensitivity_comparison.csv`, the five `ln_dsir_sel*.csv` datasets,
`ln_dsir_selection_frequency.csv`, `selector_four_way_comparison.csv`,
`paired_deltas.csv`, `diversity_comparison.csv`, `kl_reduction_comparison.csv`,
`ln_dsir_results.xlsx`, and `ln_dsir_analysis.txt`.
