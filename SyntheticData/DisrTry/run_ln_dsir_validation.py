"""Length-normalized class-conditional DSIR validation.

This keeps the original CC-DSIR artifacts untouched and adds the narrowly
specified ``log(p_target)-log(q_raw)`` divided by n-gram-count selector.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "SyntheticData" / "exp"
DISR_DIR = ROOT / "SyntheticData" / "DisrTry"
BASELINE_DIR = DISR_DIR / "outputs" / "dsir_selector_validation"
VALIDATION_DIR = EXP_DIR / "outputs" / "curation_validation"
OUTPUT_DIR = DISR_DIR / "outputs" / "ln_dsir_validation"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(DISR_DIR))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402
from run_cc_dsir_validation import (  # noqa: E402
    DSIR_MAIN,
    fit_group,
    gumbel_select,
    load_reference_artifacts,
    ngram_features,
    wpt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector", choices=["dsir_original", "dsir_length_norm"], default="dsir_length_norm")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    p.add_argument("--validation-dir", default=str(VALIDATION_DIR))
    p.add_argument("--selection-seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--hash-dimension", type=int, default=10000)
    p.add_argument("--ngrams", type=int, default=2)
    p.add_argument("--target-smoothing", type=float, default=0.0)
    p.add_argument("--evaluate-legacy400", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--merge-shards", action="store_true")
    p.add_argument("--bert-model", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def score_groups(data: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> tuple[dict[tuple[str, int], dict[str, Any]], pd.DataFrame]:
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    rows = []
    for source in ("qwen", "llama"):
        raw_source = data[f"{source}_train"]
        for label, cls in LABEL_NAMES.items():
            raw = [x for x in raw_source if int(x["label"]) == label]
            target = [x for x in data["real_reference"] if int(x["label"]) == label]
            fitted = fit_group(target, raw, args.hash_dimension, args.ngrams, args.target_smoothing)
            original = fitted["log_weights"].astype(float)
            ngrams = np.asarray([max(x[2], 1) for x in fitted["features"]], dtype=float)
            normalized = original / ngrams if args.selector == "dsir_length_norm" else original.copy()
            shifted = normalized - float(np.max(normalized))
            weights = np.exp(shifted)
            probs = weights / max(float(weights.sum()), 1e-30)
            fitted["original_log_weights"] = original
            fitted["normalized_log_weights"] = normalized
            fitted["sampling_probabilities"] = probs
            fitted["num_ngrams"] = ngrams
            groups[(source, label)] = fitted
            def stats(values: np.ndarray) -> dict[str, float]:
                return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "min": float(values.min()), "median": float(np.median(values)), "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}
            ess = float(weights.sum() ** 2 / max(float((weights ** 2).sum()), 1e-30))
            top5 = float(np.sort(probs)[-5:].sum()); top10 = float(np.sort(probs)[-10:].sum())
            token_counts = np.asarray([x[1] for x in fitted["features"]], dtype=float)
            rows.append({
                "source": source, "class": cls, "candidate_count": len(raw),
                "mean_num_tokens": float(token_counts.mean()), "std_num_tokens": float(token_counts.std(ddof=1)),
                "mean_num_ngrams": float(ngrams.mean()), "std_num_ngrams": float(ngrams.std(ddof=1)),
                "mean_original_log_weight": float(original.mean()), "std_original_log_weight": float(original.std(ddof=1)),
                "mean_normalized_log_weight": float(normalized.mean()), "std_normalized_log_weight": float(normalized.std(ddof=1)),
                "min_normalized_log_weight": float(normalized.min()), "median_normalized_log_weight": float(np.median(normalized)),
                "p90_normalized_log_weight": float(np.quantile(normalized, .90)), "p95_normalized_log_weight": float(np.quantile(normalized, .95)), "max_normalized_log_weight": float(normalized.max()),
                "ESS": ess, "ESS_ratio": ess / len(raw), "max_sampling_probability": float(probs.max()),
                "top5_probability_mass": top5, "top10_probability_mass": top10,
            })
    return groups, pd.DataFrame(rows)


def candidate_table(data: dict[str, list[dict[str, Any]]], groups: dict[tuple[str, int], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for source in ("qwen", "llama"):
        records = data[f"{source}_train"]
        for label, cls in LABEL_NAMES.items():
            raw = [x for x in records if int(x["label"]) == label]
            g = groups[(source, label)]
            for i, record in enumerate(raw):
                feat = g["features"][i][0]
                rows.append({"sample_id": record["sample_id"], "source": source, "class": cls, "label": label,
                    "text": record["text"], "num_tokens": int(g["features"][i][1]), "num_ngrams": int(g["features"][i][2]),
                    "original_log_weight": float(g["original_log_weights"][i]), "normalized_log_weight": float(g["normalized_log_weights"][i]),
                    "sampling_probability": float(g["sampling_probabilities"][i])})
    return pd.DataFrame(rows)


def length_comparison(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, cls), frame in candidates.groupby(["source", "class"]):
        x = frame.num_ngrams.to_numpy(float); yo = frame.original_log_weight.to_numpy(float); yn = frame.normalized_log_weight.to_numpy(float)
        op = float(pearsonr(x, yo).statistic); os = float(spearmanr(x, yo).statistic)
        np_ = float(pearsonr(x, yn).statistic); ns = float(spearmanr(x, yn).statistic)
        rows.append({"Source": source, "Class": cls, "Original_Pearson": op, "Original_Spearman": os, "LN_Pearson": np_, "LN_Spearman": ns, "Delta_Abs_Spearman": abs(os) - abs(ns)})
    return pd.DataFrame(rows)


def select_sets(data: dict[str, list[dict[str, Any]]], groups: dict[tuple[str, int], dict[str, Any]], seeds: list[int], selector: str) -> dict[int, list[dict[str, Any]]]:
    result = {}
    for seed in seeds:
        selected = []
        for source in ("qwen", "llama"):
            source_records = data[f"{source}_train"]
            for label in LABEL_NAMES:
                raw = [x for x in source_records if int(x["label"]) == label]
                g = groups[(source, label)]
                scores = g["normalized_log_weights"] if selector == "dsir_length_norm" else g["original_log_weights"]
                indices = gumbel_select(scores, 40, seed * 10000 + label * 100 + (0 if source == "qwen" else 1))
                selected.extend(raw[i] for i in indices)
        selected.sort(key=lambda x: (int(x["label"]), x["sample_id"]))
        result[seed] = [{**x, "selection_method": selector} for x in selected]
        if len(selected) != 320 or Counter(int(x["label"]) for x in selected) != Counter({0: 80, 1: 80, 2: 80, 3: 80}):
            raise RuntimeError(f"Invalid {selector} selection for seed {seed}")
    return result


def add_frequency(candidates: pd.DataFrame, selected: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    freq = Counter(x["sample_id"] for rows in selected.values() for x in rows)
    out = candidates[["sample_id", "source", "class", "original_log_weight", "normalized_log_weight"]].copy()
    out["ln_selection_frequency"] = out.sample_id.map(freq).fillna(0).astype(int)
    return out


def ess_comparison(data: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> pd.DataFrame:
    original_groups, _ = score_groups(data, argparse.Namespace(**{**vars(args), "selector": "dsir_original"}))
    ln_groups, _ = score_groups(data, argparse.Namespace(**{**vars(args), "selector": "dsir_length_norm"}))
    rows = []
    for source in ("qwen", "llama"):
        for label, cls in LABEL_NAMES.items():
            o, n = original_groups[(source, label)], ln_groups[(source, label)]
            def get(g: dict[str, Any]) -> tuple[float, float]:
                w = np.exp(g["normalized_log_weights"] - np.max(g["normalized_log_weights"])); e = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-30)); return e, e / len(w)
            oe, oer = get(o); ne, ner = get(n)
            rows.append({"Source": source, "Class": cls, "Original_ESS": oe, "Original_ESS_Ratio": oer, "LN_ESS": ne, "LN_ESS_Ratio": ner, "Original_Max_Probability": float(np.exp(o["normalized_log_weights"] - np.max(o["normalized_log_weights"])).max() / np.exp(o["normalized_log_weights"] - np.max(o["normalized_log_weights"])).sum()), "LN_Max_Probability": float(n["sampling_probabilities"].max())})
    return pd.DataFrame(rows)


def severe_flags(ess: pd.DataFrame, length: pd.DataFrame) -> list[str]:
    flags = []
    if bool((ess.LN_ESS < 2).any()) or bool((ess.LN_Max_Probability > .9).any()): flags.append("SEVERE_WEIGHT_COLLAPSE")
    if bool((length.LN_Spearman.abs() > .7).any()): flags.append("SEVERE_LENGTH_BIAS")
    return flags


def train_ln(selected: dict[int, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]], dev: list[dict[str, Any]], out: Path, args: argparse.Namespace) -> pd.DataFrame:
    legacy = data["real_test"] if args.evaluate_legacy400 else None
    jobs = [(s, m, rows, f"LN_DSIR_sel{s}_model{m}") for s, rows in selected.items() for m in args.model_seeds]
    if args.shard_index is not None: jobs = jobs[args.shard_index::args.num_shards]; train_dir = out / "shards" / f"shard_{args.shard_index}"
    else: train_dir = out
    train_dir.mkdir(parents=True, exist_ok=True)
    train_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False)
    rows = []
    for selection_seed, model_seed, records, run_name in jobs:
        result = validation.train_one(records, dev, legacy, None, train_dir, run_name, model_seed, train_args)
        rows.append(validation.metric_row("LN_DSIR", str(selection_seed), model_seed, result, train_dir / "training" / run_name))
    frame = pd.DataFrame(rows); frame.to_csv(train_dir / "ln_dsir_run_results.csv", index=False); return frame


def ci_rows(combined: pd.DataFrame, method: str, baseline: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        vals = []
        for seed in sorted(combined.Model_Seed.dropna().unique()):
            ln = combined[(combined.Experiment == "LN_DSIR") & (combined.Model_Seed == seed)][metric].mean()
            if baseline == "Random": ref = combined[(combined.Experiment == "E1") & (combined.Model_Seed == seed)][metric].mean()
            elif baseline == "TopK": ref = combined[(combined.Experiment == "E2") & (combined.Model_Seed == seed)][metric].iloc[0]
            else: ref = combined[(combined.Experiment == "E5") & (combined.Model_Seed == seed)][metric].mean()
            vals.append(float(ln - ref))
        a = np.asarray(vals); margin = 1.96 * a.std(ddof=1) / math.sqrt(len(a))
        rows.append({"Metric": metric, "Comparison": f"LN_DSIR_minus_{baseline}", "Delta_Mean": a.mean(), "Delta_Std": a.std(ddof=1), "CI95_Low": a.mean()-margin, "CI95_High": a.mean()+margin, "Positive_Seed_Count": int((a > 0).sum()), "N": len(a)})
    return pd.DataFrame(rows)


def main_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    specs = [("Random", "E1"), ("TopK Proximity", "E2"), ("Original CC-DSIR", "E5"), ("LN-CC-DSIR", "LN_DSIR")]
    rows = []
    for name, exp in specs:
        f = combined[combined.Experiment == exp]; row = {"Method": name, "Selector": exp, "Num_Selection_Sets": 5 if exp != "E2" else 1, "Num_Training_Runs": len(f)}
        for metric, col in [("Accuracy", "Dev_Accuracy"), ("Macro_F1", "Dev_Macro_F1"), ("World_F1", "Dev_World_F1"), ("Sports_F1", "Dev_Sports_F1"), ("Business_F1", "Dev_Business_F1"), ("SciTech_F1", "Dev_SciTech_F1")]:
            row[f"Dev_{metric}_Mean" if metric in ("Accuracy", "Macro_F1") else f"{metric}_Mean"] = f[col].mean(); row[f"Dev_{metric}_Std" if metric in ("Accuracy", "Macro_F1") else f"{metric}_Std"] = f[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics_tables(data: dict[str, list[dict[str, Any]]], selected: dict[int, list[dict[str, Any]]], original_selected: dict[int, list[dict[str, Any]]], e2: list[dict[str, Any]], args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = {"Random": {s: pd.read_csv(Path(args.validation_dir) / f"e1_random_sel{s}.csv").to_dict("records") for s in args.selection_seeds}, "TopK": {42: e2}, "Original DSIR": original_selected, "LN-DSIR": selected}
    emb_cache = Path(args.baseline_dir) / "cache"
    diversity = curation.load_or_compute_embeddings(data, out, str(ROOT / "cloud" / "model" / "roberta-base"), 32)
    # Reuse the established diagnostics implementation; it writes no state beyond embedding cache.
    from run_cc_dsir_validation import diversity_diagnostics, distribution_diagnostics
    div = diversity_diagnostics(data, sets, emb_cache)
    ln_groups, _ = score_groups(data, argparse.Namespace(**{**vars(args), "selector": "dsir_length_norm"}))
    kl, _ = distribution_diagnostics(data, sets, ln_groups, args)
    return div, kl


def run(args: argparse.Namespace) -> Path:
    if len(args.selection_seeds) != 5 or len(args.model_seeds) != 5: raise ValueError("Exactly five selection and model seeds are required")
    if args.num_shards < 1 or (args.shard_index is not None and not 0 <= args.shard_index < args.num_shards): raise ValueError("Invalid shard arguments")
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    data, dev, e2, _ = load_reference_artifacts(Path(args.validation_dir))
    ln_groups, _ = score_groups(data, argparse.Namespace(**{**vars(args), "selector": "dsir_length_norm"}))
    original_groups, _ = score_groups(data, argparse.Namespace(**{**vars(args), "selector": "dsir_original"}))
    candidates = candidate_table(data, ln_groups)
    original_candidates = candidate_table(data, original_groups)
    weight_diag = score_groups(data, argparse.Namespace(**{**vars(args), "selector": "dsir_length_norm"}))[1]
    ln_ess = ess_comparison(data, args); length = length_comparison(candidates)
    flags = severe_flags(ln_ess, length)
    weight_diag.to_csv(out / "ln_dsir_weight_diagnostics.csv", index=False); ln_ess.to_csv(out / "ess_comparison.csv", index=False); length.to_csv(out / "length_sensitivity_comparison.csv", index=False)
    selected = select_sets(data, ln_groups, args.selection_seeds, "dsir_length_norm")
    original_selected = select_sets(data, original_groups, args.selection_seeds, "dsir_original")
    for seed, rows in selected.items(): pd.DataFrame(rows).to_csv(out / f"ln_dsir_sel{seed}.csv", index=False)
    freq = add_frequency(candidates, selected); orig_freq = add_frequency(original_candidates, original_selected).rename(columns={"normalized_log_weight": "original_normalized_log_weight", "ln_selection_frequency": "original_selection_frequency"})
    freq = freq.merge(orig_freq[["sample_id", "original_selection_frequency"]], on="sample_id", how="left"); freq.to_csv(out / "ln_dsir_selection_frequency.csv", index=False)
    (out / "ln_dsir_protocol_manifest.json").write_text(json.dumps({"selector": "LN-CC-DSIR", "formula": "(log_p_target-log_q_raw)/max(num_ngrams,1)", "target": "QC train160 only", "selection_seeds": args.selection_seeds, "model_seeds": args.model_seeds, "final_holdout_evaluated": False, "severe_flags": flags}, indent=2), encoding="utf-8")
    # Diagnostics are always produced first. By default a pathological score
    # distribution stops before BERT training; --force-train is an explicit
    # override for an exploratory run.
    if args.merge_shards and flags and not args.force_train:
        raise RuntimeError("LN diagnostics still report severe pathology; use --force-train only for an explicit exploratory override")
    if args.prepare_only or (flags and not args.force_train):
        status = "\n".join(flags) if flags else "Diagnostics passed; training may proceed."
        (out / "ln_dsir_analysis.txt").write_text("Diagnostics-only status\n" + status + "\n", encoding="utf-8")
        print(f"Prepared LN-CC-DSIR diagnostics under {out}\n{status}")
        return out
    if args.merge_shards:
        shard_files = sorted((out / "shards").glob("shard_*/ln_dsir_run_results.csv"))
        if len(shard_files) == 0: raise FileNotFoundError("No LN shard result files found")
        ln_results = pd.concat([pd.read_csv(p) for p in shard_files], ignore_index=True)
        if len(ln_results) != 25: raise RuntimeError(f"Expected 25 LN runs, found {len(ln_results)}")
        baseline = pd.read_csv(Path(args.baseline_dir) / "run_results.csv"); combined = pd.concat([baseline, ln_results], ignore_index=True); combined.to_csv(out / "ln_dsir_run_results.csv", index=False)
        main = main_comparison(combined); metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]
        paired = pd.concat([ci_rows(combined, "LN_DSIR", b, metrics) for b in ("Random", "TopK", "Original DSIR")], ignore_index=True)
        div, kl = diagnostics_tables(data, selected, original_selected, e2, args, out); main.to_csv(out / "selector_four_way_comparison.csv", index=False); paired.to_csv(out / "paired_deltas.csv", index=False); div.to_csv(out / "diversity_comparison.csv", index=False); kl.to_csv(out / "kl_reduction_comparison.csv", index=False)
        with pd.ExcelWriter(out / "ln_dsir_results.xlsx", engine="openpyxl") as writer:
            main.to_excel(writer, sheet_name="Main_Comparison", index=False); main.to_excel(writer, sheet_name="Per_Class_F1", index=False); ln_ess.to_excel(writer, sheet_name="ESS_Comparison", index=False); length.to_excel(writer, sheet_name="Length_Bias", index=False); paired.to_excel(writer, sheet_name="Paired_Deltas", index=False); freq.to_excel(writer, sheet_name="Selection_Frequency", index=False); div.to_excel(writer, sheet_name="Diversity", index=False); kl.to_excel(writer, sheet_name="KL_Reduction", index=False)
        ln_acc = paired[paired.Metric == "Dev_Accuracy"]; ln_f1 = paired[paired.Metric == "Dev_Macro_F1"]
        report = ["LN-CC-DSIR validation report", f"Original median ESS={ln_ess.Original_ESS.median():.6f}", f"LN median ESS={ln_ess.LN_ESS.median():.6f}", f"Original Spearman median={length.Original_Spearman.median():.6f}; LN Spearman median={length.LN_Spearman.median():.6f}", f"Flags={flags or ['none']}", f"LN vs Random Accuracy={ln_acc[ln_acc.Comparison == 'LN_DSIR_minus_Random'].Delta_Mean.iloc[0]:.6f}; Macro-F1={ln_f1[ln_f1.Comparison == 'LN_DSIR_minus_Random'].Delta_Mean.iloc[0]:.6f}", f"LN vs TopK Accuracy={ln_acc[ln_acc.Comparison == 'LN_DSIR_minus_TopK'].Delta_Mean.iloc[0]:.6f}; Macro-F1={ln_f1[ln_f1.Comparison == 'LN_DSIR_minus_TopK'].Delta_Mean.iloc[0]:.6f}", f"LN vs Original DSIR Accuracy={ln_acc[ln_acc.Comparison == 'LN_DSIR_minus_Original DSIR'].Delta_Mean.iloc[0]:.6f}; Macro-F1={ln_f1[ln_f1.Comparison == 'LN_DSIR_minus_Original DSIR'].Delta_Mean.iloc[0]:.6f}", "Final holdout was not evaluated.", "Conclusion: objective report only."]
        (out / "ln_dsir_analysis.txt").write_text("\n".join(report) + "\n", encoding="utf-8"); return out
    selected_for_train = selected
    train_ln(selected_for_train, data, dev, out, args); return out


if __name__ == "__main__":
    run(parse_args())
