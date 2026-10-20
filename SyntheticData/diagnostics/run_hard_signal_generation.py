#!/usr/bin/env python3
"""SciTech hard-signal ablation: generate equal-budget additions and train BERT.

Uses Phase0 hard_loss assignments to form strong-only, weak-only, consensus,
and random anchor groups. One fixed Qwen generator prompt is used for all
groups. Existing frozen synthetic data is retained and each group contributes
the same number of generated SciTech examples. No selection is based on
Dev2000; it is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "SyntheticData"
PHASE0 = SYN / "diagnostics" / "outputs" / "feedback_stability"
DEFAULT_OUT = SYN / "diagnostics" / "outputs" / "hard_signal_generation"
DEFAULT_BASE = SYN / "exp" / "outputs" / "curation_validation" / "e2_uniform_topk_fixed.csv"
DEFAULT_DEV = SYN / "exp" / "outputs" / "curation_validation" / "dev2000.csv"
DEFAULT_QWEN = ROOT / "cloud" / "model" / "Qwen3-4B-Instruct-2507"
LABEL_TEXT = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}


def fixed_prompt(anchor: str) -> str:
    return (
        "You are generating a realistic AG News training item.\n"
        "Target category: SciTech\n"
        "Write one original English news article lead of 35-80 words. "
        "It must clearly belong to the target category. Return exactly one line "
        "in this format: TEXT: <article>\n"
        "Reference style example (do not copy names or numbers): " + anchor
    )


def parse_text(raw: str) -> str:
    import re
    match = re.search(r"TEXT\s*:\s*(.+)", raw, flags=re.IGNORECASE | re.DOTALL)
    text = match.group(1) if match else raw
    return re.sub(r"\s+", " ", text).strip().strip('"')


def load_qwen(model_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"local_files_only": True, "trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs.update(device_map={"": "cuda:0"}, torch_dtype=torch.float16)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).eval()
    except RuntimeError as exc:
        if not torch.cuda.is_available():
            raise
        # Some HAMI CUDA environments reject allocator warmup for large
        # device maps. Fall back to CPU rather than changing the protocol.
        print(f"CUDA generator load failed ({exc}); retrying on CPU", flush=True)
        cpu_kwargs = {"local_files_only": True, "trust_remote_code": True, "torch_dtype": torch.float16}
        model = AutoModelForCausalLM.from_pretrained(model_dir, **cpu_kwargs).to("cpu").eval()
    return tokenizer, model


@torch.inference_mode()
def generate_one(tokenizer, model, prompt: str, seed: int, max_new_tokens: int) -> str:
    from transformers import LogitsProcessor, LogitsProcessorList
    class CleanLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9).clamp(-1e9, 1e9)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rendered = prompt
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95, top_k=50, logits_processor=LogitsProcessorList([CleanLogits()]), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return parse_text(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


@torch.inference_mode()
def generate_batch(tokenizer, model, prompts: list[str], seed: int, max_new_tokens: int) -> list[str]:
    """Small-batch variant used by CPU fallback; prompt and sampler stay fixed."""
    from transformers import LogitsProcessor, LogitsProcessorList
    class CleanLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9).clamp(-1e9, 1e9)
    torch.manual_seed(seed)
    rendered = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) if getattr(tokenizer, "chat_template", None) else p for p in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95, top_k=50, logits_processor=LogitsProcessorList([CleanLogits()]), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    prompt_len = inputs["input_ids"].shape[1]
    return [parse_text(tokenizer.decode(row[prompt_len:], skip_special_tokens=True)) for row in output]


def build_groups(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    sample = pd.read_csv(PHASE0 / "sample_hardness.csv")
    sample = sample[(sample.Dataset == "reference") & (sample.label == 3)].copy()
    strong = int(args.strong_model_seed); weak = int(args.weak_model_seed)
    a = sample[sample.Model_Seed == strong].set_index("sample_id")
    b = sample[sample.Model_Seed == weak].set_index("sample_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    strong_hard = a.hard_loss.astype(bool); weak_hard = b.hard_loss.astype(bool)
    flags = pd.DataFrame({"sample_id": common, "strong_hard_loss": strong_hard.to_numpy(), "weak_hard_loss": weak_hard.to_numpy()})
    flags["signal_group"] = np.select([flags.strong_hard_loss & ~flags.weak_hard_loss, ~flags.strong_hard_loss & flags.weak_hard_loss, flags.strong_hard_loss & flags.weak_hard_loss], ["strong_model_hard", "weak_model_only_hard", "consensus_hard"], default="unassigned")
    ref = pd.read_csv(PHASE0 / "reference_samples.csv").set_index("sample_id")
    flags = flags.join(ref[["source_index", "text", "label", "reference", "region"]], on="sample_id")
    rng = np.random.default_rng(args.anchor_seed)
    groups = {}
    for name in ("strong_model_hard", "weak_model_only_hard", "consensus_hard"):
        pool = flags[flags.signal_group == name]
        if len(pool) < args.anchors_per_group:
            raise RuntimeError(f"{name} has only {len(pool)} anchors; requested {args.anchors_per_group}")
        chosen = pool.iloc[np.sort(rng.choice(len(pool), size=args.anchors_per_group, replace=False))].copy()
        chosen["signal_group"] = name
        groups[name] = chosen.to_dict("records")
    random_pool = flags[flags.signal_group == "unassigned"]
    if len(random_pool) < args.anchors_per_group:
        raise RuntimeError(f"Random pool has only {len(random_pool)} anchors")
    chosen = random_pool.iloc[np.sort(rng.choice(len(random_pool), size=args.anchors_per_group, replace=False))].copy()
    chosen["signal_group"] = "random"
    groups["random"] = chosen.to_dict("records")
    pd.concat([pd.DataFrame(v) for v in groups.values()], ignore_index=True).to_csv(Path(args.output_dir) / "hard_signal_anchor_groups.csv", index=False)
    return groups


def generate_groups(groups, args: argparse.Namespace, out: Path) -> dict[str, list[dict[str, Any]]]:
    if args.reuse_pool:
        return reuse_pool_groups(groups, args, out)
    existing = set(pd.read_csv(args.base_train).text.astype(str).str.strip()) | set(pd.read_csv(args.dev_csv).text.astype(str).str.strip())
    generated = {}
    tokenizer, model = load_qwen(Path(args.generator_model))
    batch_size = args.generation_batch_size if next(model.parameters()).device.type == "cpu" else 1
    for group, anchors in groups.items():
        rows = []
        for anchor_i, anchor in enumerate(anchors):
            prompt = fixed_prompt(anchor["text"])
            for replicate in range(args.outputs_per_anchor):
                seed = args.generation_seed + list(groups).index(group) * 100000 + anchor_i * 100 + replicate
                text = ""
                for attempt in range(args.max_attempts):
                    if batch_size == 1:
                        text = generate_one(tokenizer, model, prompt, seed + attempt, args.max_new_tokens)
                    else:
                        text = generate_batch(tokenizer, model, [prompt], seed + attempt, args.max_new_tokens)[0]
                    if text and text.strip() not in existing and len(text.split()) >= 15:
                        break
                if not text or text.strip() in existing:
                    raise RuntimeError(f"Could not create unique generation for {group}/{anchor_i}/{replicate}")
                existing.add(text.strip())
                rows.append({"sample_id": f"hardgen_{group}_{anchor_i:03d}_{replicate:02d}", "text": text, "label": 3, "class": "SciTech", "generator": "qwen", "signal_group": group, "anchor_sample_id": anchor["sample_id"], "anchor_region": anchor["region"], "prompt_version": "fixed_sciTech_anchor_v1", "generation_seed": seed})
        generated[group] = rows
        pd.DataFrame(rows).to_csv(out / f"generated_{group}.csv", index=False)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return generated


def reuse_pool_groups(groups, args: argparse.Namespace, out: Path) -> dict[str, list[dict[str, Any]]]:
    """Build equal-budget signal groups from a frozen historical generator pool.

    The historical pool has no anchor IDs, so this is deliberately a loose
    ablation: signal controls semantic selection from the fixed Qwen pool,
    while no new text is generated.  Selection is deterministic and groups
    are disjoint to prevent leakage between conditions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    pool_path = Path(args.candidate_pool).resolve()
    pool = pd.read_csv(pool_path)
    required = {"sample_id", "source", "label", "text"}
    missing = required - set(pool.columns)
    if missing:
        raise RuntimeError(f"Candidate pool missing columns: {sorted(missing)}")
    pool = pool[(pool.source.astype(str).str.lower() == args.generator_source.lower()) & (pool.label.astype(int) == 3)].copy()
    if args.pool_prompt_version:
        if "prompt_version" not in pool.columns:
            raise RuntimeError("--pool-prompt-version requested but candidate pool has no prompt_version column")
        pool = pool[pool.prompt_version.astype(str) == args.pool_prompt_version].copy()
    if pool.empty:
        raise RuntimeError(f"No label=3 rows for source={args.generator_source} in {pool_path}")
    forbidden = set(pd.read_csv(args.base_train).text.astype(str).str.strip()) | set(pd.read_csv(args.dev_csv).text.astype(str).str.strip())
    pool = pool[~pool.text.astype(str).str.strip().isin(forbidden)].drop_duplicates("text").reset_index(drop=True)
    budget = args.anchors_per_group * args.outputs_per_anchor
    if len(pool) < budget * 4:
        raise RuntimeError(f"Pool has {len(pool)} usable rows, need {budget * 4}")

    group_names = ["strong_model_hard", "weak_model_only_hard", "consensus_hard"]
    # Fit on both sides so rare SciTech terms are represented consistently.
    anchor_texts = [str(x["text"]) for name in group_names for x in groups[name]]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(anchor_texts + pool.text.astype(str).tolist())
    anchor_matrix = matrix[: len(anchor_texts)]
    pool_matrix = matrix[len(anchor_texts):]
    scores = {}
    offset = 0
    for name in group_names:
        n = len(groups[name])
        scores[name] = cosine_similarity(pool_matrix, anchor_matrix[offset:offset + n]).max(axis=1)
        offset += n

    rng = np.random.default_rng(args.pool_seed)
    available = np.ones(len(pool), dtype=bool)
    selected: dict[str, list[dict[str, Any]]] = {}
    # Draw the random baseline before signal selection so it is uniform over
    # the complete eligible pool rather than a residual after ranking.
    random_take = rng.choice(np.flatnonzero(available), size=budget, replace=False)
    chosen = pool.iloc[np.sort(random_take)].copy()
    chosen["signal_group"] = "random"
    chosen["selection_score"] = np.nan
    chosen["selection_method"] = "uniform_random_from_fixed_prompt_qwen_scitech_pool"
    chosen["anchor_sample_id"] = ""
    selected["random"] = chosen.to_dict("records")
    available[random_take] = False

    # Reserve semantically closest records for signal groups.  Consensus is
    # selected first because it is the smallest/more specific target region.
    for name in ["consensus_hard", "strong_model_hard", "weak_model_only_hard"]:
        order = np.argsort(-scores[name], kind="stable")
        take = []
        for i in order:
            i = int(i)
            if available[i]:
                take.append(i)
                if len(take) == budget:
                    break
        if len(take) < budget:
            raise RuntimeError(f"Could not select {budget} disjoint rows for {name}")
        available[take] = False
        chosen = pool.iloc[take].copy()
        chosen["signal_group"] = name
        chosen["selection_score"] = scores[name][take]
        chosen["selection_method"] = "tfidf_max_cosine_to_signal_anchors"
        group_anchor_matrix = anchor_matrix[
            sum(len(groups[g]) for g in group_names[:group_names.index(name)]):
            sum(len(groups[g]) for g in group_names[:group_names.index(name) + 1])
        ]
        local = cosine_similarity(pool_matrix[take], group_anchor_matrix).argmax(axis=1)
        chosen["anchor_sample_id"] = [groups[name][int(j)]["sample_id"] for j in local]
        selected[name] = chosen.to_dict("records")

    all_rows = pd.concat([pd.DataFrame(v) for v in selected.values()], ignore_index=True)
    all_rows.to_csv(out / "reused_pool_group_samples.csv", index=False)
    for name, rows in selected.items():
        pd.DataFrame(rows).to_csv(out / f"generated_{name}.csv", index=False)
    (out / "reused_pool_selection_scores.csv").write_text(
        all_rows[["sample_id", "signal_group", "selection_score", "selection_method"]].to_csv(index=False),
        encoding="utf-8",
    )
    return selected


def train_groups(groups, args: argparse.Namespace, generated, out: Path) -> None:
    sys.path.insert(0, str(SYN / "dimp"))
    from phase1_trainers import Phase1Trainer
    base = pd.read_csv(args.base_train).to_dict("records")
    dev = pd.read_csv(args.dev_csv).to_dict("records")
    model_dir = args.bert_model
    rows = []
    for group, additions in generated.items():
        train = base + additions
        for seed in args.model_seeds:
            run_name = f"{group}_model{seed}"
            result = Phase1Trainer(model_name=model_dir, num_labels=4, loss_type="ce").train(train, dev, quality_scores=None, output_dir=out / "training", run_name=run_name, seed=seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, save_checkpoints=False)
            rows.append({"Signal_Group": group, "ModelSeed": seed, "Train_Count": len(train), "Generated_Count": len(additions), "Dev_Accuracy": result["metrics"]["eval_accuracy"], "Dev_Macro_F1": result["metrics"]["eval_macro_f1"], "Run_Dir": str(out / "training" / run_name)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "hard_signal_generation_run_results.csv", index=False)
    baseline = frame.groupby("Signal_Group")[["Dev_Accuracy", "Dev_Macro_F1"]].mean().rename(columns=lambda x: x + "_Mean")
    std = frame.groupby("Signal_Group")[["Dev_Accuracy", "Dev_Macro_F1"]].std(ddof=1).rename(columns=lambda x: x + "_Std")
    summary = baseline.join(std).reset_index()
    summary.to_csv(out / "hard_signal_generation_summary.csv", index=False)
    base_group = summary[summary.Signal_Group == "random"]
    mode_line = (f"Mode: reuse existing frozen {args.generator_source} pool ({args.pool_prompt_version}); TF-IDF selection to anchors; no new generation." if args.reuse_pool else "Mode: fresh Qwen generation with a fixed prompt template.")
    lines = ["SciTech hard-signal generation ablation", "", mode_line, f"Strong model seed: {args.strong_model_seed}; weak model seed: {args.weak_model_seed}; anchors/group: {args.anchors_per_group}; outputs/anchor: {args.outputs_per_anchor}; budget/group: {args.anchors_per_group * args.outputs_per_anchor}.", "Dev2000 is evaluation-only; it does not define anchors or prompts.", "", summary.to_string(index=False)]
    if args.reuse_pool:
        lines += ["", "Caveat: the historical pool has no anchor IDs. This loose ablation tests signal-conditioned selection from a fixed Qwen pool, not freshly generated anchor-conditioned text."]
    if not base_group.empty:
        rand_acc = float(base_group.Dev_Accuracy_Mean.iloc[0]); rand_f1 = float(base_group.Dev_Macro_F1_Mean.iloc[0])
        lines += ["", "Gain relative to Random:"]
        for _, r in summary.iterrows():
            lines.append(f"{r.Signal_Group}: Accuracy={r.Dev_Accuracy_Mean - rand_acc:+.6f}; Macro-F1={r.Dev_Macro_F1_Mean - rand_f1:+.6f}")
    (out / "hard_signal_generation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    groups = build_groups(args)
    manifest = {"strong_model_seed": args.strong_model_seed, "weak_model_seed": args.weak_model_seed, "anchor_seed": args.anchor_seed, "anchors_per_group": args.anchors_per_group, "outputs_per_anchor": args.outputs_per_anchor, "generator": str(Path(args.generator_model).resolve()), "prompt_version": args.pool_prompt_version if args.reuse_pool else "fixed_sciTech_anchor_v1", "groups": {k: [x["sample_id"] for x in v] for k, v in groups.items()}, "no_new_data_selection_from_dev": True, "generation_mode": "reused_existing_pool" if args.reuse_pool else "fresh_generation", "candidate_pool": str(Path(args.candidate_pool).resolve()) if args.reuse_pool else None, "generator_source": args.generator_source if args.reuse_pool else "qwen", "pool_prompt_version": args.pool_prompt_version if args.reuse_pool else None, "pool_seed": args.pool_seed if args.reuse_pool else None, "loose_protocol_caveat": "Historical Qwen outputs have no anchor IDs; signal controls TF-IDF selection from a frozen pool, not fresh anchor-conditioned generation." if args.reuse_pool else None}
    (out / "hard_signal_generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    generated = generate_groups(groups, args, out)
    train_groups(groups, args, generated, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(DEFAULT_OUT)); p.add_argument("--base-train", default=str(DEFAULT_BASE)); p.add_argument("--dev-csv", default=str(DEFAULT_DEV)); p.add_argument("--generator-model", default=str(DEFAULT_QWEN)); p.add_argument("--bert-model", default=os.environ.get("BERT_MODEL", "bert-base-uncased")); p.add_argument("--strong-model-seed", type=int, default=43); p.add_argument("--weak-model-seed", type=int, default=46); p.add_argument("--anchor-seed", type=int, default=20261003); p.add_argument("--generation-seed", type=int, default=20261004); p.add_argument("--anchors-per-group", type=int, default=15); p.add_argument("--outputs-per-anchor", type=int, default=4); p.add_argument("--generation-batch-size", type=int, default=4); p.add_argument("--max-attempts", type=int, default=4); p.add_argument("--max-new-tokens", type=int, default=96); p.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=256); p.add_argument("--reuse-pool", action="store_true"); p.add_argument("--candidate-pool", default=str(ROOT / "cloud" / "model" / "extra_data" / "candidate_pool" / "outputs" / "expanded_candidate_pool.csv")); p.add_argument("--generator-source", default="qwen"); p.add_argument("--pool-prompt-version", default="synthetic_agnews.build_prompt.v1"); p.add_argument("--pool-seed", type=int, default=20261005)
    main(p.parse_args())
