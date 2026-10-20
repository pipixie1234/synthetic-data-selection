"""Llama-specific selector confirmation: four fixed datasets x three seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "SyntheticData" / "exp"
CAR_DIR = ROOT / "SyntheticData" / "CaRTry"
VAL_DIR = EXP / "outputs" / "curation_validation"
LN_DIR = ROOT / "SyntheticData" / "DisrTry" / "outputs" / "ln_dsir_validation"
OLD_ABLATION = CAR_DIR / "outputs" / "llama_length_ablation"
OUT = CAR_DIR / "outputs" / "llama_selector_confirmation"
sys.path.insert(0, str(EXP))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402

SELECTORS = ["Random40", "TopK40", "LN-DSIR40", "RA-CaR40"]
SEEDS = [42, 43, 44]
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}
LEXICAL = {"Sports": ["Lakers", "Celtics", "LeBron", "NBA", "football", "soccer", "tennis"], "SciTech": ["quantum", "quantum computing", "breakthrough", "Google", "chip", "AI", "Microsoft"]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT)); p.add_argument("--validation-dir", default=str(VAL_DIR)); p.add_argument("--ln-dir", default=str(LN_DIR)); p.add_argument("--car-dir", default=str(CAR_DIR / "outputs" / "car_selector_validation"))
    p.add_argument("--model-seeds", type=int, nargs="+", default=SEEDS); p.add_argument("--prepare-only", action="store_true"); p.add_argument("--evaluate-legacy400", action="store_true")
    p.add_argument("--bert-model", default="bert-base-uncased"); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def ids_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(x["sample_id"] for x in rows)).encode()).hexdigest()


def load_frame(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict("records")


def make_fixed_sets(args: argparse.Namespace, data: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    val = Path(args.validation_dir); ln_dir = Path(args.ln_dir); car_dir = Path(args.car_dir)
    qwen_top = [x for x in load_frame(val / "e2_uniform_topk_fixed.csv") if x["source"] == "qwen"]
    old_random = [x for x in load_frame(OLD_ABLATION / "llama_ablation_Random40.csv") if x["source"] == "llama"]
    old_top = [x for x in load_frame(OLD_ABLATION / "llama_ablation_TopK40.csv") if x["source"] == "llama"]
    ln = [x for x in load_frame(ln_dir / "ln_dsir_sel42.csv") if x["source"] == "llama"]
    car = [x for x in load_frame(car_dir / "car_sel42.csv") if x["source"] == "llama"]
    sources = {"Random40": old_random, "TopK40": old_top, "LN-DSIR40": ln, "RA-CaR40": car}
    qids = {x["sample_id"] for x in qwen_top}
    if len(qwen_top) != 160 or any(len(v) != 160 for v in sources.values()): raise RuntimeError("Invalid fixed subset size")
    for name, rows in sources.items():
        if any(x["sample_id"].startswith("real_test_") or x["sample_id"].startswith("dev_") for x in rows): raise RuntimeError(f"Leakage in {name}")
        if {x["sample_id"] for x in rows} & qids: raise RuntimeError(f"Qwen/Llama collision in {name}")
        if any(sum(x["source"] == "llama" and int(x["label"]) == y for x in rows) != 40 for y in range(4)): raise RuntimeError(f"Bad Llama allocation in {name}")
    if {x["sample_id"] for x in qwen_top} != {x["sample_id"] for x in load_frame(OLD_ABLATION / "llama_ablation_Random40.csv") if x["source"] == "qwen"}: raise RuntimeError("QWEN_DATA_MISMATCH")
    if {x["sample_id"] for x in qwen_top} != {x["sample_id"] for x in load_frame(OLD_ABLATION / "llama_ablation_TopK40.csv") if x["source"] == "qwen"}: raise RuntimeError("QWEN_DATA_MISMATCH")
    result = {}
    for name, llama in sources.items():
        rows = qwen_top + llama
        rows = []
        for x in qwen_top + llama:
            item = dict(x); item["selector"] = name; item["class"] = LABEL_NAMES[int(x["label"])]; rows.append(item)
        rows.sort(key=lambda x: (int(x["label"]), x["sample_id"]))
        if len(rows) != 320 or len({x["sample_id"] for x in rows}) != 320: raise RuntimeError(f"Invalid {name} dataset")
        result[name] = rows
        pd.DataFrame(rows)[["sample_id", "source", "class", "label", "text", "selector"]].to_csv(Path(args.output_dir) / f"{name}_dataset.csv", index=False)
    return result


def integrity(args: argparse.Namespace, sets: dict[str, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for selector, records in sets.items():
        qids = {x["sample_id"] for x in records if x["source"] == "qwen"}; lids = {x["sample_id"] for x in records if x["source"] == "llama"}
        rows.append({"selector": selector, "total": len(records), "qwen": len(qids), "llama": len(lids), "unique_ids": len({x["sample_id"] for x in records}), "id_hash": ids_hash(records), "qwen_id_hash": hashlib.sha256("\n".join(sorted(qids)).encode()).hexdigest(), "all_class_balanced": all(sum(x["source"] == s and int(x["label"]) == y for x in records) == 40 for s in ("qwen", "llama") for y in range(4)), "leakage": any(x["sample_id"].startswith(("real_test_", "dev_", "official_test_")) for x in records)})
    frame = pd.DataFrame(rows); frame.to_csv(Path(args.output_dir) / "llama_selector_fixed_dataset_manifest.csv", index=False)
    if frame.leakage.any() or not frame.all_class_balanced.all() or frame.total.ne(320).any() or frame.qwen.nunique() != 1 or frame.qwen_id_hash.nunique() != 1: raise RuntimeError("Dataset integrity or QWEN_DATA_MISMATCH failure")
    return frame


def score_margin_and_proximity(args: argparse.Namespace, sets: dict[str, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]], scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    llama_scores = scores[scores.source == "llama"].copy(); llama_scores["num_tokens"] = llama_scores.text.map(lambda x: len(str(x).split()))
    score_map = scores.set_index("sample_id")["top5_real_cosine_mean"].to_dict()
    margins = []; prox = []
    for label, cls in LABEL_NAMES.items():
        f = llama_scores[llama_scores.label == label].sort_values(["top5_real_cosine_mean", "sample_id"], ascending=[False, True]).reset_index(drop=True); vals = f.top5_real_cosine_mean.to_numpy(float); margins.append({"Class": cls, "score_rank1": vals[0], "score_rank10": vals[9], "score_rank20": vals[19], "score_rank40": vals[39], "score_rank41": vals[40], "score_rank60": vals[59], "score_rank80": vals[79], "boundary_margin": vals[39] - vals[40], "total_range": vals[0] - vals[79], "normalized_boundary_margin": (vals[39] - vals[40]) / max(vals[0] - vals[79], 1e-12)})
        for selector, rows in sets.items():
            selected = [x for x in rows if x["source"] == "llama" and int(x["label"]) == label]; scores_sel = np.asarray([float(score_map[x["sample_id"]]) for x in selected]); ranks = f.set_index("sample_id").loc[[x["sample_id"] for x in selected]].index
            rank_map = {sid: i + 1 for i, sid in enumerate(f.sample_id)}; rr = np.asarray([rank_map[x["sample_id"]] for x in selected])
            prox.append({"Selector": selector, "Class": cls, "mean_proximity": scores_sel.mean(), "std_proximity": scores_sel.std(ddof=1), "median_proximity": np.median(scores_sel), "p10_proximity": np.quantile(scores_sel, .1), "p90_proximity": np.quantile(scores_sel, .9), "min_proximity": scores_sel.min(), "max_proximity": scores_sel.max(), "mean_proximity_rank": rr.mean()})
    return pd.DataFrame(margins), pd.DataFrame(prox), llama_scores


def anchors_and_lexical(args: argparse.Namespace, sets: dict[str, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = Path(args.validation_dir) / "cache"; train = torch.load(cache / "llama_train_embeddings.pt", weights_only=False); real = torch.load(cache / "real_reference_embeddings.pt", weights_only=False); ids = train["sample_ids"]; rid = real["sample_ids"]; te = np.asarray(train["embeddings"], dtype=float); re = np.asarray(real["embeddings"], dtype=float); te /= np.maximum(np.linalg.norm(te, axis=1, keepdims=True), 1e-12); re /= np.maximum(np.linalg.norm(re, axis=1, keepdims=True), 1e-12)
    anchors = []; lexical = []
    for selector, rows in sets.items():
        for label, cls in LABEL_NAMES.items():
            pool_indices = [i for i, x in enumerate(data["llama_train"]) if int(x["label"]) == label]; real_indices = [i for i, x in enumerate(real["labels"]) if int(x) == label]; real_ids = [rid[i] for i in real_indices]; sims = te[pool_indices] @ re[real_indices].T; top5 = {ids[pool_indices[j]]: [real_ids[k] for k in np.argsort(sims[j])[::-1][:5]] for j in range(len(pool_indices))}; selected_ids = {x["sample_id"] for x in rows if x["source"] == "llama" and int(x["label"]) == label}; hits = Counter(a for sid, aa in top5.items() if sid in selected_ids for a in aa); unique = len(hits); total = sum(hits.values()); probs = np.asarray(list(hits.values()), float) / max(total, 1); entropy = float(-sum(p * np.log(p) for p in probs if p > 0)); anchors.append({"Selector": selector, "Class": cls, "unique_real_anchors_covered": unique, "real_reference_class_count": len(real_ids), "anchor_coverage_ratio": unique / max(1, len(real_ids)), "anchor_hit_entropy": entropy})
            vocab = LEXICAL.get(cls, []); texts = [str(x["text"]).lower() for x in rows if x["source"] == "llama" and int(x["label"]) == label]
            for phrase in vocab:
                count = sum(phrase.lower() in t for t in texts); lexical.append({"Selector": selector, "Class": cls, "Pattern": phrase, "Count": count, "Proportion": count / max(1, len(texts))})
    return pd.DataFrame(anchors), pd.DataFrame(lexical)


def diversity_overlap(args: argparse.Namespace, sets: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = Path(args.validation_dir) / "cache" / "llama_train_embeddings.pt"; loaded = torch.load(cache, weights_only=False); lookup = dict(zip(loaded["sample_ids"], np.asarray(loaded["embeddings"], dtype=float))); div = []
    for selector, rows in sets.items():
        for label, cls in LABEL_NAMES.items():
            group = [x for x in rows if x["source"] == "llama" and int(x["label"]) == label]; vec = np.asarray([lookup[x["sample_id"]] for x in group]); vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12); sim = vec @ vec.T; vals = sim[np.triu_indices(len(group), 1)]; nn = np.max(sim - np.eye(len(group)), axis=1); div.append({"Selector": selector, "Class": cls, "pairwise_cosine_mean": vals.mean(), "nearest_neighbor_cosine_mean": nn.mean()})
    overlap = []
    for a in SELECTORS:
        for b in SELECTORS:
            if SELECTORS.index(a) >= SELECTORS.index(b): continue
            for label, cls in LABEL_NAMES.items():
                ai = {x["sample_id"] for x in sets[a] if x["source"] == "llama" and int(x["label"]) == label}; bi = {x["sample_id"] for x in sets[b] if x["source"] == "llama" and int(x["label"]) == label}; overlap.append({"Selector_A": a, "Selector_B": b, "Class": cls, "intersection_count": len(ai & bi), "jaccard": len(ai & bi) / max(1, len(ai | bi))})
    return pd.DataFrame(div), pd.DataFrame(overlap)


def train(args: argparse.Namespace, sets: dict[str, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    out = Path(args.output_dir); dev = pd.read_csv(Path(args.validation_dir) / "dev2000.csv").to_dict("records"); legacy = data["real_test"] if args.evaluate_legacy400 else None; train_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False); rows = []
    for selector, records in sets.items():
        for seed in args.model_seeds:
            run_name = f"{selector}_model{seed}"
            # The previous one-seed ablation used the same frozen files and
            # trainer configuration for these two exact runs. Reuse its
            # results instead of retraining them.
            if seed == 42 and selector in {"Random40", "TopK40"}:
                old_name = f"LlamaAblation_{selector}_model42"
                result = validation.train_one(records, dev, legacy, None, OLD_ABLATION, old_name, seed, train_args)
                run_dir = OLD_ABLATION / "training" / old_name
            else:
                result = validation.train_one(records, dev, legacy, None, out, run_name, seed, train_args)
                run_dir = out / "training" / run_name
            rows.append(validation.metric_row(selector, "fixed", seed, result, run_dir))
    frame = pd.DataFrame(rows); frame.to_csv(out / "llama_selector_12runs.csv", index=False); return frame


def analysis_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = results.groupby("Experiment").agg(N_Runs=("Model_Seed", "count"), Dev_Accuracy_Mean=("Dev_Accuracy", "mean"), Dev_Accuracy_Std=("Dev_Accuracy", "std"), Dev_Macro_F1_Mean=("Dev_Macro_F1", "mean"), Dev_Macro_F1_Std=("Dev_Macro_F1", "std"), World_F1_Mean=("Dev_World_F1", "mean"), Sports_F1_Mean=("Dev_Sports_F1", "mean"), Business_F1_Mean=("Dev_Business_F1", "mean"), SciTech_F1_Mean=("Dev_SciTech_F1", "mean")).reset_index().rename(columns={"Experiment": "Selector"})
    rows = []
    for comparator, a in [("Random - TopK", "Random40"), ("LN-DSIR - TopK", "LN-DSIR40"), ("RA-CaR - TopK", "RA-CaR40")]:
        for metric in ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]:
            vals = [float(results[(results.Experiment == a) & (results.Model_Seed == s)][metric].iloc[0] - results[(results.Experiment == "TopK40") & (results.Model_Seed == s)][metric].iloc[0]) for s in args_global_seeds]
            rows.append({"Comparator": comparator, "Metric": metric, "Seed42_Delta": vals[0], "Seed43_Delta": vals[1], "Seed44_Delta": vals[2], "Mean_Delta": np.mean(vals), "Std_Delta": np.std(vals, ddof=1), "Positive_Seeds": int(sum(x > 0 for x in vals))})
    paired = pd.DataFrame(rows); per = paired[paired.Metric.str.contains("World|Sports|Business|SciTech")].copy(); return summary, paired, per


args_global_seeds = SEEDS


def main() -> None:
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); args_global_seeds[:] = args.model_seeds
    if args.model_seeds != SEEDS: raise ValueError("This protocol requires model seeds 42,43,44")
    data = curation.load_frozen_data(); sets = make_fixed_sets(args, data); manifest = integrity(args, sets, data); scores = pd.read_csv(Path(args.validation_dir) / "sample_proximity_scores.csv"); margins, prox, llama_scores = score_margin_and_proximity(args, sets, data, scores); anchors, lexical = anchors_and_lexical(args, sets, data); div, overlap = diversity_overlap(args, sets)
    margins.to_csv(out / "llama_topk_score_margin.csv", index=False); anchors.to_csv(out / "llama_anchor_coverage.csv", index=False); prox.to_csv(out / "llama_proximity_distribution.csv", index=False); lexical.to_csv(out / "llama_lexical_modes.csv", index=False); div.to_csv(out / "llama_semantic_diversity.csv", index=False); overlap.to_csv(out / "llama_selector_overlap.csv", index=False); (out / "dataset_manifest.json").write_text(manifest.to_json(orient="records", indent=2), encoding="utf-8")
    if args.prepare_only: print(f"Prepared confirmation artifacts under {out}"); return
    results = train(args, sets, data); summary, paired, per = analysis_tables(results); summary.to_csv(out / "llama_selector_summary.csv", index=False); paired.to_csv(out / "llama_selector_paired_deltas.csv", index=False)
    with pd.ExcelWriter(out / "llama_selector_confirmation.xlsx", engine="openpyxl") as w:
        results.to_excel(w, sheet_name="Main_Results", index=False); summary.to_excel(w, sheet_name="Selector_Summary", index=False); paired.to_excel(w, sheet_name="Paired_Deltas", index=False); per.to_excel(w, sheet_name="Per_Class_F1", index=False); margins.to_excel(w, sheet_name="Score_Margins", index=False); anchors.to_excel(w, sheet_name="Anchor_Coverage", index=False); prox.to_excel(w, sheet_name="Proximity_Distribution", index=False); div.to_excel(w, sheet_name="Semantic_Diversity", index=False); lexical.to_excel(w, sheet_name="Lexical_Modes", index=False); manifest.to_excel(w, sheet_name="Dataset_Manifest", index=False)
    rt = paired[(paired.Comparator == "Random - TopK") & (paired.Metric == "Dev_Accuracy")].iloc[0]; rf = paired[(paired.Comparator == "Random - TopK") & (paired.Metric == "Dev_Macro_F1")].iloc[0]; report = ["Llama selector confirmation report", f"Random vs TopK Accuracy deltas: seed42={rt.Seed42_Delta:.6f}, seed43={rt.Seed43_Delta:.6f}, seed44={rt.Seed44_Delta:.6f}; mean={rt.Mean_Delta:.6f}; positive={int(rt.Positive_Seeds)}/3", f"Random vs TopK Macro-F1 deltas: seed42={rf.Seed42_Delta:.6f}, seed43={rf.Seed43_Delta:.6f}, seed44={rf.Seed44_Delta:.6f}; mean={rf.Mean_Delta:.6f}; positive={int(rf.Positive_Seeds)}/3", f"TopK score boundary normalized margin mean={margins.normalized_boundary_margin.mean():.6f}", f"TopK anchor coverage ratio mean={anchors[anchors.Selector == 'TopK40'].anchor_coverage_ratio.mean():.6f}", f"TopK semantic pairwise cosine mean={div[div.Selector == 'TopK40'].pairwise_cosine_mean.mean():.6f}", "Final holdout was not evaluated."]
    (out / "llama_selector_confirmation.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(results.to_string(index=False)); print("\n".join(report))


if __name__ == "__main__":
    main()
