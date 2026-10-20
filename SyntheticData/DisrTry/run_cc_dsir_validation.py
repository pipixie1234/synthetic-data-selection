"""Class-Conditional DSIR selector validation for the frozen AG News protocol."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "SyntheticData" / "exp"
DISR_DIR = ROOT / "SyntheticData" / "DisrTry"
VALIDATION_DIR = EXP_DIR / "outputs" / "curation_validation"
OUTPUT_DIR = DISR_DIR / "outputs" / "dsir_selector_validation"
DSIR_MAIN = ROOT / "dsir-main"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(DSIR_MAIN))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402
# dsir-main is available as source but is not necessarily pip-installed.
import importlib.metadata as importlib_metadata  # noqa: E402
_original_version = importlib_metadata.version
importlib_metadata.version = lambda name: "1.0.3" if name == "data-selection" else _original_version(name)
from data_selection.hashed_ngram_dsir import get_ngram_counts, wpt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector", choices=["dsir_original", "dsir_length_norm"], default="dsir_original")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--validation-dir", default=str(VALIDATION_DIR))
    p.add_argument("--baseline-dir", default=str(OUTPUT_DIR))
    p.add_argument("--selection-seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--hash-dimension", type=int, default=10000)
    p.add_argument("--ngrams", type=int, default=2)
    p.add_argument("--tokenizer", choices=["wordpunct"], default="wordpunct")
    p.add_argument("--target-smoothing", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=5)
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


def ngram_features(text: str, hash_dimension: int, ngrams: int) -> tuple[np.ndarray, int, int]:
    """Use dsir-main's exact lower/wordpunct/hash/unigram-bigram extractor."""
    tokens = wpt.tokenize(text.lower())
    features = get_ngram_counts(text, n=ngrams, num_buckets=hash_dimension, tokenizer=wpt.tokenize)
    return features.astype(np.float64), len(tokens), int(features.sum())


def fit_group(target: list[dict[str, Any]], raw: list[dict[str, Any]], hash_dimension: int, ngrams: int, smoothing: float) -> dict[str, Any]:
    target_counts = np.zeros(hash_dimension, dtype=np.float64)
    raw_counts = np.zeros(hash_dimension, dtype=np.float64)
    raw_feature_rows = []
    for record in target:
        target_counts += ngram_features(record["text"], hash_dimension, ngrams)[0]
    for record in raw:
        features, num_tokens, num_ngrams = ngram_features(record["text"], hash_dimension, ngrams)
        raw_counts += features
        raw_feature_rows.append((features, num_tokens, num_ngrams))
    if target_counts.sum() <= 0 or raw_counts.sum() <= 0:
        raise RuntimeError("DSIR target/raw feature counts are empty")
    target_probs = target_counts + smoothing
    target_probs /= target_probs.sum()
    raw_probs = raw_counts / raw_counts.sum()
    # This is the reference implementation's exact numerical convention.
    log_diff = np.log(target_probs + 1e-8) - np.log(raw_probs + 1e-8)
    log_weights = np.asarray([float(np.dot(log_diff, row[0])) for row in raw_feature_rows])
    return {
        "target_counts": target_counts,
        "raw_counts": raw_counts,
        "target_probs": target_probs,
        "raw_probs": raw_probs,
        "log_diff": log_diff,
        "features": raw_feature_rows,
        "log_weights": log_weights,
    }


def stable_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / max(float(weights.sum()), 1e-30)


def gumbel_select(log_weights: np.ndarray, count: int, seed: int) -> list[int]:
    if count > len(log_weights):
        raise ValueError("Cannot select more DSIR samples than candidates")
    rng = np.random.default_rng(seed)
    scores = log_weights + rng.gumbel(size=len(log_weights))
    return np.argpartition(-scores, count - 1)[:count].tolist()


def build_datasets(data: dict[str, list[dict[str, Any]]], selection_seeds: list[int], args: argparse.Namespace) -> tuple[dict[int, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for source in ("qwen", "llama"):
        source_records = data[f"{source}_train"]
        for label in LABEL_NAMES:
            raw = [x for x in source_records if int(x["label"]) == label]
            target = [x for x in data["real_reference"] if int(x["label"]) == label]
            groups[(source, label)] = fit_group(target, raw, args.hash_dimension, args.ngrams, args.target_smoothing)

    candidate_rows = []
    selected_by_seed: dict[int, list[dict[str, Any]]] = {}
    for source in ("qwen", "llama"):
        source_records = data[f"{source}_train"]
        for label in LABEL_NAMES:
            raw = [x for x in source_records if int(x["label"]) == label]
            fitted = groups[(source, label)]
            probs = stable_weights(fitted["log_weights"])
            for i, record in enumerate(raw):
                _, num_tokens, num_ngrams = fitted["features"][i]
                candidate_rows.append({
                    "sample_id": record["sample_id"], "source": source, "class": LABEL_NAMES[label], "label": label,
                    "text": record["text"], "num_tokens": int(num_tokens), "num_ngrams": int(num_ngrams),
                    "log_p_target": float(np.dot(np.log(fitted["target_probs"] + 1e-8), fitted["features"][i][0])),
                    "log_q_raw": float(np.dot(np.log(fitted["raw_probs"] + 1e-8), fitted["features"][i][0])),
                    "log_importance_weight": float(fitted["log_weights"][i]),
                    "normalized_log_weight": float(fitted["log_weights"][i] - np.max(fitted["log_weights"])),
                    "normalized_sampling_probability": float(probs[i]),
                })
    candidates = pd.DataFrame(candidate_rows)
    frequency = Counter()
    for seed in selection_seeds:
        selected = []
        for source in ("qwen", "llama"):
            source_records = data[f"{source}_train"]
            for label in LABEL_NAMES:
                raw = [x for x in source_records if int(x["label"]) == label]
                idx = gumbel_select(groups[(source, label)]["log_weights"], 40, seed * 10000 + label * 100 + (0 if source == "qwen" else 1))
                selected.extend(raw[i] for i in idx)
        selected.sort(key=lambda x: (int(x["label"]), x["sample_id"]))
        selected_rows = []
        for row in selected:
            item = dict(row)
            item["proximity_score"] = np.nan
            item["selection_method"] = "class_conditional_dsir"
            item.pop("original_index", None)
            selected_rows.append(item)
            frequency[item["sample_id"]] += 1
        if len(selected_rows) != 320 or Counter(x["label"] for x in selected_rows) != Counter({0: 80, 1: 80, 2: 80, 3: 80}):
            raise RuntimeError(f"E5 selection {seed} failed size/class checks")
        selected_by_seed[seed] = selected_rows
    frequency_rows = []
    for row in candidate_rows:
        frequency_rows.append({"sample_id": row["sample_id"], "source": row["source"], "class": row["class"], "importance_weight": float(np.exp(row["normalized_log_weight"])), "selection_frequency": int(frequency[row["sample_id"]])})
    return selected_by_seed, candidates, pd.DataFrame(frequency_rows), groups


def length_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, cls), frame in candidates.groupby(["source", "class"]):
        x = frame["num_tokens"].to_numpy(dtype=float)
        y = frame["log_importance_weight"].to_numpy(dtype=float)
        if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            pearson = float(pearsonr(x, y).statistic)
            spearman = float(spearmanr(x, y).statistic)
        else:
            pearson = spearman = np.nan
        rows.append({"source": source, "class": cls, "Pearson_length_log_weight": pearson, "Spearman_length_log_weight": spearman, "warning": bool(np.isfinite(spearman) and abs(spearman) > 0.5)})
    return pd.DataFrame(rows)


def weight_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, cls), frame in candidates.groupby(["source", "class"]):
        weights = np.exp(frame["normalized_log_weight"].to_numpy(dtype=float))
        ess = float(weights.sum() ** 2 / max((weights ** 2).sum(), 1e-30))
        rows.append({"source": source, "class": cls, "candidate_count": len(frame), "mean_log_weight": frame.log_importance_weight.mean(), "std_log_weight": frame.log_importance_weight.std(ddof=1), "min_log_weight": frame.log_importance_weight.min(), "median_log_weight": frame.log_importance_weight.median(), "p90_log_weight": frame.log_importance_weight.quantile(.90), "p95_log_weight": frame.log_importance_weight.quantile(.95), "max_log_weight": frame.log_importance_weight.max(), "ESS": ess, "ESS_ratio": ess / len(frame)})
    return pd.DataFrame(rows)


def load_reference_artifacts(validation_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    data = curation.load_frozen_data()
    dev = pd.read_csv(validation_dir / "dev2000.csv").to_dict("records")
    e2 = pd.read_csv(validation_dir / "e2_uniform_topk_fixed.csv").to_dict("records")
    return data, dev, e2, pd.read_csv(validation_dir / "sample_proximity_scores.csv")


def selected_map(selected_by_seed: dict[int, list[dict[str, Any]]]) -> dict[int, set[str]]:
    return {seed: {x["sample_id"] for x in rows} for seed, rows in selected_by_seed.items()}


def kl_divergence(target: np.ndarray, other: np.ndarray, eps: float = 1e-8) -> float:
    p = target + eps; p = p / p.sum()
    q = other + eps; q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def distribution_diagnostics(data: dict[str, list[dict[str, Any]]], selected_sets: dict[str, dict[int, list[dict[str, Any]]]], groups: dict[tuple[str, int], dict[str, Any]], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    kl_rows = []
    mmd_rows = []
    emb = curation.load_or_compute_embeddings(data, OUTPUT_DIR, str(ROOT / "cloud" / "model" / "roberta-base"), 32)
    for source in ("qwen", "llama"):
        for label, cls in LABEL_NAMES.items():
            raw = [x for x in data[f"{source}_train"] if int(x["label"]) == label]
            target = [x for x in data["real_reference"] if int(x["label"]) == label]
            fitted = groups[(source, label)]
            raw_counts = fitted["raw_counts"]
            target_counts = fitted["target_counts"]
            for method, method_sets in selected_sets.items():
                for seed, rows in method_sets.items():
                    selected = [x for x in rows if int(x["label"]) == label and x["source"] == source]
                    selected_counts = np.zeros(args.hash_dimension, dtype=np.float64)
                    for x in selected:
                        selected_counts += ngram_features(x["text"], args.hash_dimension, args.ngrams)[0]
                    raw_kl = kl_divergence(target_counts, raw_counts)
                    selected_kl = kl_divergence(target_counts, selected_counts)
                    kl_rows.append({"method": method, "source": source, "class": cls, "selection_seed": seed, "KL_target_raw": raw_kl, "KL_target_selected": selected_kl, "KL_reduction": raw_kl - selected_kl})
                    real_emb = emb["real_reference"]; source_emb = emb[f"{source}_train"]
                    real_x = real_emb["embeddings"][np.asarray(real_emb["labels"]) == label]
                    raw_x = source_emb["embeddings"][np.asarray(source_emb["labels"]) == label]
                    selected_ids = {x["sample_id"] for x in selected}
                    selected_idx = [i for i, sid in enumerate(source_emb["sample_ids"]) if sid in selected_ids]
                    selected_x = source_emb["embeddings"][selected_idx]
                    n = min(len(real_x), len(raw_x))
                    rng = np.random.default_rng(2026 + label)
                    raw_x_n = raw_x[rng.choice(len(raw_x), n, replace=False)]
                    raw_mmd = curation.unbiased_mmd(raw_x_n, real_x[:n], curation.median_bandwidth(raw_x_n, real_x[:n]))
                    selected_mmd = curation.unbiased_mmd(selected_x, real_x[:len(selected_x)], curation.median_bandwidth(selected_x, real_x[:len(selected_x)]))
                    mmd_rows.append({"method": method, "source": source, "class": cls, "selection_seed": seed, "MMD_target_raw": raw_mmd, "MMD_target_selected": selected_mmd, "MMD_reduction": raw_mmd - selected_mmd, "diagnostic_only": True})
    return pd.DataFrame(kl_rows), pd.DataFrame(mmd_rows)


def diversity_diagnostics(data: dict[str, list[dict[str, Any]]], selected_sets: dict[str, dict[int, list[dict[str, Any]]]], embedding_cache_dir: Path) -> pd.DataFrame:
    embeddings = {}
    for source in ("qwen_train", "llama_train"):
        loaded = torch.load(embedding_cache_dir / f"{source}_embeddings.pt", weights_only=False)
        embeddings.update(dict(zip(loaded["sample_ids"], loaded["embeddings"])))
    rows = []
    for method, sets in selected_sets.items():
        for seed, records in sets.items():
            for source in ("qwen", "llama"):
                for label, cls in LABEL_NAMES.items():
                    group = [x for x in records if x["source"] == source and int(x["label"]) == label]
                    vectors = np.asarray([embeddings[x["sample_id"]] for x in group], dtype=np.float64)
                    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                    sim = vectors @ vectors.T
                    vals = sim[np.triu_indices(len(group), 1)]
                    nn = np.max(sim - np.eye(len(group)), axis=1)
                    unigrams = []; bigrams = []
                    for x in group:
                        toks = wpt.tokenize(x["text"].lower()); unigrams.extend(toks); bigrams.extend(zip(toks, toks[1:]))
                    rows.append({"method": method, "selection_seed": seed, "source": source, "class": cls, "pairwise_cosine_mean": vals.mean(), "pairwise_cosine_median": np.median(vals), "pairwise_cosine_p90": np.quantile(vals, .9), "nearest_neighbor_mean": nn.mean(), "unique_unigram_ratio": len(set(unigrams)) / max(len(unigrams), 1), "unique_bigram_ratio": len(set(bigrams)) / max(len(bigrams), 1), "exact_duplicate_count": len(group) - len({x["text"] for x in group})})
    return pd.DataFrame(rows)


def overlap_diagnostics(e2: list[dict[str, Any]], selected_by_seed: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    topk_ids = {x["sample_id"] for x in e2}
    rows = []
    for source in ("qwen", "llama"):
        for label, cls in LABEL_NAMES.items():
            top_ids = {x["sample_id"] for x in e2 if x["source"] == source and int(x["label"]) == label}
            vals = []
            for seed, rows_seed in selected_by_seed.items():
                dsir_ids = {x["sample_id"] for x in rows_seed if x["source"] == source and int(x["label"]) == label}
                vals.append(curation.jaccard(dsir_ids, top_ids) if hasattr(curation, "jaccard") else len(dsir_ids & top_ids) / len(dsir_ids | top_ids))
            rows.append({"source": source, "class": cls, "TopK_DSIR_Jaccard_Mean": np.mean(vals), "TopK_DSIR_Jaccard_Std": np.std(vals, ddof=1), "selection_count": len(vals)})
    return pd.DataFrame(rows)


def train_e5(selected_by_seed: dict[int, list[dict[str, Any]]], dev: list[dict[str, Any]], data: dict[str, list[dict[str, Any]]], output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    legacy = data["real_test"] if args.evaluate_legacy400 else None
    jobs = [(seed, model_seed, rows, f"E5_dsir_sel{seed}_model{model_seed}") for seed, rows in selected_by_seed.items() for model_seed in args.model_seeds]
    if args.shard_index is not None:
        jobs = jobs[args.shard_index::args.num_shards]
        train_dir = output_dir / "shards" / f"shard_{args.shard_index}"
    else:
        train_dir = output_dir
    train_dir.mkdir(parents=True, exist_ok=True)
    train_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False)
    rows = []
    for selection_seed, model_seed, selected, run_name in jobs:
        result = validation.train_one(selected, dev, legacy, None, train_dir, run_name, model_seed, train_args)
        rows.append(validation.metric_row("E5", str(selection_seed), model_seed, result, train_dir / "training" / run_name))
    frame = pd.DataFrame(rows)
    frame.to_csv(train_dir / "run_results.csv", index=False)
    return frame


def paired_summary(all_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]
    e1 = all_results[all_results.Experiment == "E1"]
    e2 = all_results[all_results.Experiment == "E2"]
    e5 = all_results[all_results.Experiment == "E5"]
    rows = []
    for model_seed in sorted(e2.Model_Seed.unique()):
        random_rows = e1[e1.Model_Seed == model_seed]
        top_row = e2[e2.Model_Seed == model_seed].iloc[0]
        dsir_rows = e5[e5.Model_Seed == model_seed]
        for metric in metrics:
            random_mean = random_rows[metric].mean(); dsir_mean = dsir_rows[metric].mean(); top = top_row[metric]
            rows.append({"Model_Seed": model_seed, "Metric": metric, "RandomMean": random_mean, "TopK": top, "DSIRMean": dsir_mean, "DSIR_minus_Random": dsir_mean - random_mean, "DSIR_minus_TopK": dsir_mean - top})
    paired = pd.DataFrame(rows)
    summaries = []
    for metric in metrics:
        sub = paired[paired.Metric == metric]
        for comparison in ("DSIR_minus_Random", "DSIR_minus_TopK"):
            vals = sub[comparison].to_numpy(float); margin = 1.96 * vals.std(ddof=1) / math.sqrt(len(vals))
            summaries.append({"Metric": metric, "Comparison": comparison, "Delta_Mean": vals.mean(), "Delta_Std": vals.std(ddof=1), "CI95_Low": vals.mean() - margin, "CI95_High": vals.mean() + margin, "Positive_Count": int((vals > 0).sum()), "N": len(vals)})
    win_rows = []
    for metric in ["Dev_Accuracy", "Dev_Macro_F1"]:
        wins_random = []; wins_top = []
        for model_seed in sorted(e2.Model_Seed.unique()):
            random_rows = e1[e1.Model_Seed == model_seed][metric].to_numpy(float)
            dsir_rows = e5[e5.Model_Seed == model_seed][metric].to_numpy(float)
            top_value = float(e2[e2.Model_Seed == model_seed].iloc[0][metric])
            wins_random.extend((dsir_rows[:, None] > random_rows[None, :]).ravel().astype(int))
            wins_top.extend((dsir_rows > top_value).astype(int))
        win_rows.extend([{ "Metric": metric, "Comparison": "DSIR_vs_Random", "Wins": int(sum(wins_random)), "Comparisons": len(wins_random), "Win_Rate": float(np.mean(wins_random))}, {"Metric": metric, "Comparison": "DSIR_vs_TopK", "Wins": int(sum(wins_top)), "Comparisons": len(wins_top), "Win_Rate": float(np.mean(wins_top))}])
    return paired, pd.DataFrame(summaries), pd.DataFrame(win_rows)


def merge_and_analyze(output_dir: Path, validation_dir: Path, args: argparse.Namespace) -> None:
    shard_paths = sorted((output_dir / "shards").glob("shard_*/run_results.csv"))
    if not shard_paths:
        raise FileNotFoundError("No E5 shard results found")
    e5 = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    if len(e5) != 25 or len(e5[["Selection_Seed", "Model_Seed"]].drop_duplicates()) != 25:
        raise RuntimeError(f"Expected 25 E5 runs, found {len(e5)}")
    baseline = pd.read_csv(validation_dir / "run_results.csv")
    all_results = pd.concat([baseline, e5], ignore_index=True)
    all_results.to_csv(output_dir / "run_results.csv", index=False)
    selected_by_seed, candidates, frequency, groups = load_prepared(output_dir, args)
    data, dev, e2, scores = load_reference_artifacts(validation_dir)
    paired, paired_stats, wins = paired_summary(all_results)
    method_rows = []
    metric_map = {"Accuracy": "Dev_Accuracy", "Macro_F1": "Dev_Macro_F1", "World_F1": "Dev_World_F1", "Sports_F1": "Dev_Sports_F1", "Business_F1": "Dev_Business_F1", "SciTech_F1": "Dev_SciTech_F1"}
    for method, nsets, frame in [("Random", 5, baseline[baseline.Experiment == "E1"]), ("TopK", 1, baseline[baseline.Experiment == "E2"]), ("DSIR", 5, e5)]:
        row = {"Method": method, "Num_Selection_Sets": nsets, "Num_Training_Runs": len(frame)}
        for out_name, col in metric_map.items():
            row[f"{out_name}_Mean"] = frame[col].mean(); row[f"{out_name}_Std"] = frame[col].std(ddof=1)
        method_rows.append(row)
    main = pd.DataFrame(method_rows)
    per_class = main[["Method", "World_F1_Mean", "World_F1_Std", "Sports_F1_Mean", "Sports_F1_Std", "Business_F1_Mean", "Business_F1_Std", "SciTech_F1_Mean", "SciTech_F1_Std"]]
    weight_stats = weight_diagnostics(candidates)
    length_stats = length_diagnostics(candidates)
    selection_freq = frequency
    overlap = overlap_diagnostics(e2, selected_by_seed)
    selected_sets = {"Random": {int(seed): pd.read_csv(validation_dir / f"e1_random_sel{seed}.csv").to_dict("records") for seed in [41, 42, 43, 44, 45]}, "TopK": {42: e2}, "DSIR": selected_by_seed}
    diversity = diversity_diagnostics(data, selected_sets, validation_dir / "cache")
    kl, mmd = distribution_diagnostics(data, selected_sets, groups, args)
    sheets = {"Main_Comparison": main, "Per_Class_F1": per_class, "DSIR_Selection_Seeds": e5, "Paired_Deltas": paired_stats, "Weight_Diagnostics": weight_stats, "Selection_Frequency": selection_freq, "TopK_DSIR_Overlap": overlap, "Diversity": diversity, "KL_Reduction": kl, "MMD_Reduction": mmd, "Length_Sensitivity": length_stats}
    with pd.ExcelWriter(output_dir / "dsir_selector_results.xlsx", engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    ar = paired_stats[(paired_stats.Metric == "Dev_Accuracy") & (paired_stats.Comparison == "DSIR_minus_Random")].iloc[0]
    mr = paired_stats[(paired_stats.Metric == "Dev_Macro_F1") & (paired_stats.Comparison == "DSIR_minus_Random")].iloc[0]
    at = paired_stats[(paired_stats.Metric == "Dev_Accuracy") & (paired_stats.Comparison == "DSIR_minus_TopK")].iloc[0]
    mt = paired_stats[(paired_stats.Metric == "Dev_Macro_F1") & (paired_stats.Comparison == "DSIR_minus_TopK")].iloc[0]
    report = [
        "Class-Conditional DSIR selector validation (descriptive report)",
        f"DSIR vs Random: Accuracy delta={ar.Delta_Mean:.6f}; Macro-F1 delta={mr.Delta_Mean:.6f}",
        f"DSIR vs TopK: Accuracy delta={at.Delta_Mean:.6f}; Macro-F1 delta={mt.Delta_Mean:.6f}",
        f"Positive paired model-seed deltas DSIR>Random: Accuracy={int(ar.Positive_Count)}/5; Macro-F1={int(mr.Positive_Count)}/5",
        f"Positive paired model-seed deltas DSIR>TopK: Accuracy={int(at.Positive_Count)}/5; Macro-F1={int(mt.Positive_Count)}/5",
        f"DSIR win rates: {wins.to_dict(orient='records')}",
        f"TopK/DSIR overlap mean={overlap.TopK_DSIR_Jaccard_Mean.mean():.6f}, std={overlap.TopK_DSIR_Jaccard_Mean.std(ddof=1):.6f}",
        f"Length sensitivity warnings={int(length_stats.warning.sum())}/8",
        "Final holdout was not evaluated by this script.",
        "Interpretation: exploratory only; conclusion must be one of strong positive signal, mixed signal, or no clear advantage.",
    ]
    (output_dir / "dsir_selector_analysis.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def load_prepared(output_dir: Path, args: argparse.Namespace) -> tuple[dict[int, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    selected_by_seed = {int(p.stem.replace("e5_dsir_sel", "")): pd.read_csv(p).to_dict("records") for p in output_dir.glob("e5_dsir_sel*.csv")}
    candidates = pd.read_csv(output_dir / "dsir_candidate_weights.csv")
    frequency = pd.read_csv(output_dir / "dsir_selection_frequency.csv")
    data = curation.load_frozen_data()
    groups = {}
    for source in ("qwen", "llama"):
        for label in LABEL_NAMES:
            raw = [x for x in data[f"{source}_train"] if int(x["label"]) == label]
            target = [x for x in data["real_reference"] if int(x["label"]) == label]
            groups[(source, label)] = fit_group(target, raw, args.hash_dimension, args.ngrams, args.target_smoothing)
    return selected_by_seed, candidates, frequency, groups


def run(args: argparse.Namespace) -> Path:
    if args.selector == "dsir_length_norm":
        # Keep the historical CC-DSIR directory immutable when the new mode
        # is selected without an explicit output path.
        if args.output_dir == str(OUTPUT_DIR):
            args.output_dir = str(DISR_DIR / "outputs" / "ln_dsir_validation")
        from run_ln_dsir_validation import run as run_length_normalized
        return run_length_normalized(args)
    if len(args.selection_seeds) != 5 or len(args.model_seeds) != 5:
        raise ValueError("This protocol requires five selection seeds and five model seeds")
    if args.merge_shards:
        merge_and_analyze(Path(args.output_dir), Path(args.validation_dir), args)
        return Path(args.output_dir)
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = Path(args.validation_dir).resolve()
    data, dev, e2, _ = load_reference_artifacts(validation_dir)
    selected_by_seed, candidates, frequency, groups = build_datasets(data, args.selection_seeds, args)
    candidates.to_csv(output_dir / "dsir_candidate_weights.csv", index=False)
    frequency.to_csv(output_dir / "dsir_selection_frequency.csv", index=False)
    weight_diagnostics(candidates).to_csv(output_dir / "dsir_weight_summary.csv", index=False)
    length_diagnostics(candidates).to_csv(output_dir / "length_sensitivity.csv", index=False)
    for seed, rows in selected_by_seed.items():
        pd.DataFrame(rows).to_csv(output_dir / f"e5_dsir_sel{seed}.csv", index=False)
    manifest = {"selector": "Class-Conditional DSIR (CC-DSIR)", "dsir_source": str(DSIR_MAIN), "real_reference": "qc_train", "real_reference_ids": [x["sample_id"] for x in data["real_reference"]], "synthetic_candidate_ids": {s: [x["sample_id"] for x in data[f"{s}_train"]] for s in ("qwen", "llama")}, "dev2000_ids": json.loads((validation_dir / "dev2000_ids.json").read_text()), "hash_dimension": args.hash_dimension, "ngrams": args.ngrams, "tokenizer": args.tokenizer, "target_smoothing": args.target_smoothing, "raw_smoothing": "none (reference implementation)", "log_epsilon": 1e-8, "sampling": "log_importance_weight + Gumbel(0,1), top-40 within each source x class", "selection_seeds": args.selection_seeds, "model_seeds": args.model_seeds, "selected_ids": {str(seed): [x["sample_id"] for x in rows] for seed, rows in selected_by_seed.items()}, "final_holdout_evaluated": False}
    (output_dir / "dsir_protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {
        "implementation_path": str(DSIR_MAIN),
        "reused_functions": ["data_selection.hashed_ngram_dsir.get_ngram_counts", "data_selection.hashed_ngram_dsir.hash_buckets", "data_selection.base.DSIR.resample algorithm (Gumbel top-k)"],
        "feature_hashing": "SHA-256 UTF-8 hash modulo num_buckets",
        "feature_construction": "lowercase wordpunct tokens; unigrams and contiguous bigrams",
        "hash_dimension": args.hash_dimension,
        "target_raw_fit": "token-count n-gram distributions; target uses all class reference records, raw uses all source x class candidates",
        "smoothing": {"target_laplace_smoothing": args.target_smoothing, "log_probability_epsilon": 1e-8, "raw_distribution": "no additive smoothing in dsir-main"},
        "importance_space": "log-space dot product: log_p_target - log_q_raw",
        "sampling": "weighted sampling without replacement via log_weight + Gumbel(0,1), then top-40",
        "fixed_length_assumption": False,
        "length_diagnostic": "num_tokens, num_ngrams, Pearson/Spearman length vs log weight; warning if abs(Spearman)>0.5",
        "selection_seed_control": "numpy default_rng(seed * 10000 + class * 100 + source_offset)",
    }
    (output_dir / "dsir_implementation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared CC-DSIR artifacts under {output_dir}")
        return output_dir
    train_e5(selected_by_seed, dev, data, output_dir, args)
    return output_dir


if __name__ == "__main__":
    run(parse_args())
