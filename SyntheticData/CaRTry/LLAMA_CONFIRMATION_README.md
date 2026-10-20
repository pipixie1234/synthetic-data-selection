# Llama selector confirmation

This experiment fixes the exact Qwen TopK160 subset and compares four fixed
Llama160 subsets across model seeds 42, 43, and 44:

- `Random40`
- `TopK40`
- `LN-DSIR40` using selection seed 42
- `RA-CaR40` using cluster seed 42

The output directory is `outputs/llama_selector_confirmation/`. Original DSIR,
Shortest40, and new length matching are excluded. Final holdout is never
evaluated.

Prepare and run:

```bash
cd /root/workspace
python -u SyntheticData/CaRTry/run_llama_selector_confirmation.py \
  --prepare-only --evaluate-legacy400
```

Then train the 12 runs on one GPU:

```bash
tmux new-session -d -s llama_confirm -c /root/workspace \
  'env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH python -u SyntheticData/CaRTry/run_llama_selector_confirmation.py --bert-model /root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 --evaluate-legacy400 > SyntheticData/CaRTry/llama_selector_confirmation.log 2>&1'
```

`Random40/model42` and `TopK40/model42` are reused from the previous exact
ablation when their cached result is present. The other ten runs are trained.
