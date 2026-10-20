#!/usr/bin/env python3
"""Phase 0: feedback stability of hard regions across RealReference160 draws.

The synthetic train file is fixed. Five disjoint balanced RealReference160
sets are sampled from AG News train, five BERT seeds are trained once, and all
25 reference/model cells are evaluated post-hoc. Dev2000 is audit-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "SyntheticData"
PROTOCOL = SYN / "SmaExp" / "outputs" / "protocol"
DEFAULT_SYNTHETIC = SYN / "exp" / "outputs" / "curation_validation" / "e2_uniform_topk_fixed.csv"
DEFAULT_DEV_DIR = SYN / "exp" / "outputs" / "curation_validation"
DEFAULT_OUT = SYN / "diagnostics" / "outputs" / "feedback_stability"
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DATASETS_CACHE = Path(os.environ.get("HF_DATASETS_CACHE", HF_HOME / "datasets"))
TRAIN_ARROW = DATASETS_CACHE.rglob("ag_news-train.arrow")
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}


def load_arrow_train() -> list[dict[str, Any]]:
    matches = list(TRAIN_ARROW)
    if not matches:
        raise FileNotFoundError("Cached ag_news-train.arrow was not found")
    ds = Dataset.from_file(str(matches[0]))
    return [{"sample_id": f"ag_train_{i}", "source_index": i, "text": str(row["text"]), "label": int(row["label"])} for i, row in enumerate(ds)]


def load_real_exclusions() -> set[int]:
    rows = [json.loads(x) for x in (PROTOCOL / "real_train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {int(x["source_index"]) for x in rows}


def load_dev(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path / "dev2000.csv")
    if len(frame) != 2000 or frame.groupby("label").size().to_dict() != {0: 500, 1: 500, 2: 500, 3: 500}:
        raise ValueError("Dev2000 must be the fixed balanced 2,000-row audit split")
    return [{"sample_id": str(r.sample_id), "source_index": int(r.original_dataset_index), "text": str(r.text), "label": int(r.label)} for r in frame.itertuples()]


def make_references(train: list[dict[str, Any]], dev: list[dict[str, Any]], seeds: list[int]) -> dict[str, list[dict[str, Any]]]:
    excluded = load_real_exclusions() | {int(x["source_index"]) for x in dev}
    by_class = {label: [x for x in train if x["label"] == label and x["source_index"] not in excluded] for label in LABEL_NAMES}
    refs: dict[str, list[dict[str, Any]]] = {}
    remaining = {label: list(rows) for label, rows in by_class.items()}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen: list[dict[str, Any]] = []
        for label in LABEL_NAMES:
            if len(remaining[label]) < 40:
                raise RuntimeError("Not enough disjoint AG News train rows for five RealReference160 sets")
            idx = rng.choice(len(remaining[label]), size=40, replace=False)
            take = {int(i) for i in idx}
            chosen.extend(remaining[label][i] for i in sorted(take))
            remaining[label] = [x for i, x in enumerate(remaining[label]) if i not in take]
        refs[f"ref_{seed}"] = sorted(chosen, key=lambda x: (x["label"], x["source_index"]))
    return refs


def load_synthetic(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Synthetic file must contain {sorted(required)}")
    rows = [{"sample_id": str(getattr(r, "sample_id", f"synthetic_{i}")), "text": str(r.text), "label": int(r.label)} for i, r in enumerate(frame.itertuples())]
    counts = frame.groupby("label").size().to_dict()
    if set(counts) != set(LABEL_NAMES) or len(set(counts.values())) != 1:
        raise ValueError(f"Synthetic training set must be balanced across four classes: {counts}")
    return rows


def fit_regions(refs: dict[str, list[dict[str, Any]]], cache_path: Path, k: int, pca_dim: int, seed: int) -> tuple[dict[int, MiniBatchKMeans], PCA, dict[int, np.ndarray]]:
    emb = np.load(cache_path, mmap_mode="r")
    all_rows = sum(refs.values(), [])
    idx = np.asarray([x["source_index"] for x in all_rows], dtype=np.int64)
    pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized").fit(np.asarray(emb[idx]))
    reduced = pca.transform(np.asarray(emb[idx])).astype(np.float32)
    models: dict[int, MiniBatchKMeans] = {}
    centroids: dict[int, np.ndarray] = {}
    for label in LABEL_NAMES:
        local = np.where(np.asarray([x["label"] for x in all_rows]) == label)[0]
        km = MiniBatchKMeans(n_clusters=k, random_state=seed + label, n_init=10, batch_size=128, max_iter=100).fit(reduced[local])
        models[label] = km
        centroids[label] = km.cluster_centers_.astype(np.float32)
    return models, pca, centroids


def assign_regions(rows: list[dict[str, Any]], emb: np.ndarray, pca: PCA, models: dict[int, MiniBatchKMeans]) -> list[str]:
    idx = np.asarray([x["source_index"] for x in rows], dtype=np.int64)
    reduced = pca.transform(np.asarray(emb[idx])).astype(np.float32)
    return [f"{label}:{int(models[label].predict(reduced[i:i + 1])[0])}" for i, label in enumerate([x["label"] for x in rows])]


def evaluate_model(model_dir: Path, rows: list[dict[str, Any]], batch_size: int, max_length: int, device: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    out: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            block = rows[start:start + batch_size]
            toks = tokenizer([x["text"] for x in block], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            labels = torch.tensor([x["label"] for x in block], device=device)
            logits = model(**toks).logits
            probs = logits.softmax(-1)
            loss = F.cross_entropy(logits, labels, reduction="none")
            pred = logits.argmax(-1)
            correct_prob = probs.gather(1, labels[:, None]).squeeze(1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            for i, row in enumerate(block):
                out.append({"sample_id": row["sample_id"], "source_index": row.get("source_index", -1), "label": row["label"], "loss": float(loss[i].cpu()), "correct": int(pred[i].item() == row["label"]), "confidence": float(correct_prob[i].cpu()), "margin": float(margin[i].cpu())})
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pd.DataFrame(out)


def hard_flags(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["hard_loss"] = False
    for label, idx in frame.groupby("label").groups.items():
        cutoff = float(frame.loc[idx, "loss"].quantile(1.0 - quantile))
        frame.loc[idx, "hard_loss"] = frame.loc[idx, "loss"] >= cutoff
    frame["hard_error"] = frame["correct"] == 0
    return frame


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    train = load_arrow_train(); dev = load_dev(Path(args.dev_dir)); refs = make_references(train, dev, args.reference_seeds); synthetic = load_synthetic(Path(args.synthetic_train))
    emb = np.load(args.embedding_cache, mmap_mode="r")
    models, pca, centroids = fit_regions(refs, Path(args.embedding_cache), args.clusters, args.pca_dim, args.region_seed)
    ref_rows = []
    for name, rows in refs.items():
        regions = assign_regions(rows, emb, pca, models)
        ref_rows.extend([{**row, "reference": name, "region": region} for row, region in zip(rows, regions)])
    ref_rows_by_name = {
        name: [row for row in ref_rows if row["reference"] == name]
        for name in refs
    }
    dev_regions = assign_regions(dev, emb, pca, models)
    dev_rows = [{**row, "region": region} for row, region in zip(dev, dev_regions)]
    (out / "reference_manifest.json").write_text(json.dumps({"reference_seeds": args.reference_seeds, "model_seeds": args.model_seeds, "per_class": 40, "disjoint": True, "synthetic_train": str(Path(args.synthetic_train).resolve()), "dev2000_audit_only": str((Path(args.dev_dir) / "dev2000.csv").resolve())}, indent=2), encoding="utf-8")
    pd.DataFrame(ref_rows).to_csv(out / "reference_samples.csv", index=False)

    sys.path.insert(0, str(SYN / "dimp"))
    from phase1_trainers import Phase1Trainer
    all_metrics = []
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    train_root = out / "training"
    for model_seed in args.model_seeds:
        run_name = f"synthetic_fixed_model{model_seed}"
        final_dir = train_root / run_name / "final"
        result_path = train_root / run_name / "results.json"
        if not final_dir.exists() or not result_path.exists():
            Phase1Trainer(model_name=args.bert_model, num_labels=4, loss_type="ce").train(synthetic, dev, quality_scores=None, output_dir=train_root, run_name=run_name, seed=model_seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, save_checkpoints=True)
        for ref_name, rows in ref_rows_by_name.items():
            metrics = hard_flags(evaluate_model(final_dir, [{**x} for x in rows], args.eval_batch_size, args.max_length, device), args.hard_quantile)
            metrics["Reference"] = ref_name; metrics["Model_Seed"] = model_seed; metrics["Dataset"] = "reference"
            metrics["region"] = [x["region"] for x in rows]
            all_metrics.append(metrics)
        metrics = hard_flags(evaluate_model(final_dir, [{**x} for x in dev_rows], args.eval_batch_size, args.max_length, device), args.hard_quantile)
        metrics["Reference"] = "all_refs_projection"; metrics["Model_Seed"] = model_seed; metrics["Dataset"] = "dev2000"; metrics["region"] = [x["region"] for x in dev_rows]
        all_metrics.append(metrics)
    samples = pd.concat(all_metrics, ignore_index=True)
    samples.to_csv(out / "sample_hardness.csv", index=False)

    within = []
    for ref, frame in samples[samples.Dataset == "reference"].groupby("Reference"):
        for metric in ("hard_loss", "hard_error"):
            sets = {int(seed): set(g.loc[g[metric].astype(bool), "sample_id"]) for seed, g in frame.groupby("Model_Seed")}
            for a, b in combinations(sorted(sets), 2): within.append({"Reference": ref, "Metric": metric, "Model_A": a, "Model_B": b, "Jaccard": jaccard(sets[a], sets[b])})
    within_df = pd.DataFrame(within); within_df.to_csv(out / "within_reference_model_variance.csv", index=False)

    region_rows = []
    for (dataset, ref, seed), frame in samples.groupby(["Dataset", "Reference", "Model_Seed"]):
        grouped = frame.groupby(["label", "region"], as_index=False).agg(Hard_Loss_Rate=("hard_loss", "mean"), Error_Rate=("hard_error", "mean"), Mean_Loss=("loss", "mean"), N=("sample_id", "size"))
        grouped["Dataset"] = dataset; grouped["Reference"] = ref; grouped["Model_Seed"] = seed; region_rows.append(grouped)
    region_df = pd.concat(region_rows, ignore_index=True); region_df.to_csv(out / "region_hardness.csv", index=False)

    across = []
    ref_region = region_df[region_df.Dataset == "reference"]
    for seed, frame in ref_region.groupby("Model_Seed"):
        piv = frame.pivot(index="Reference", columns="region", values="Hard_Loss_Rate").fillna(0.0)
        for a, b in combinations(piv.index, 2):
            across.append({"Model_Seed": seed, "Reference_A": a, "Reference_B": b, "Hard_Region_Spearman": float(spearmanr(piv.loc[a], piv.loc[b]).statistic), "Top3_Region_Jaccard": jaccard(set(piv.loc[a].nlargest(3).index), set(piv.loc[b].nlargest(3).index))})
    pd.DataFrame(across).to_csv(out / "across_reference_variance.csv", index=False)

    validity = []
    ref_region = region_df[region_df.Dataset == "reference"]
    dev_region = region_df[region_df.Dataset == "dev2000"]
    for seed in args.model_seeds:
        for ref_name in refs:
            r = ref_region[(ref_region.Model_Seed == seed) & (ref_region.Reference == ref_name)].set_index("region")
            d = dev_region[dev_region.Model_Seed == seed].set_index("region").reindex(r.index).fillna(0.0)
            validity.append({"Model_Seed": seed, "Reference": ref_name, "Hard_Region_Spearman_Ref_vs_Dev": float(spearmanr(r["Hard_Loss_Rate"], d["Error_Rate"]).statistic), "Top3_Ref_vs_Dev_Jaccard": jaccard(set(r["Hard_Loss_Rate"].nlargest(3).index), set(d["Error_Rate"].nlargest(3).index))})
    pd.DataFrame(validity).to_csv(out / "dev2000_target_validity.csv", index=False)
    report = {"reference_seeds": args.reference_seeds, "model_seeds": args.model_seeds, "within_mean_jaccard": float(within_df.Jaccard.mean()), "across_mean_spearman": float(pd.DataFrame(across).Hard_Region_Spearman.mean()), "dev2000_mean_spearman": float(pd.DataFrame(validity).Hard_Region_Spearman_Ref_vs_Dev.mean()), "dev2000_is_audit_only": True}
    (out / "feedback_stability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(DEFAULT_OUT)); p.add_argument("--synthetic-train", default=str(DEFAULT_SYNTHETIC)); p.add_argument("--dev-dir", default=str(DEFAULT_DEV_DIR)); p.add_argument("--embedding-cache", default=str(SYN / "real_reference_uncertainty" / "outputs" / "ag_news_train_roberta_cls_embeddings.npy")); p.add_argument("--bert-model", default="bert-base-uncased"); p.add_argument("--reference-seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105]); p.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--eval-batch-size", type=int, default=64); p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=256); p.add_argument("--hard-quantile", type=float, default=0.2); p.add_argument("--clusters", type=int, default=8); p.add_argument("--pca-dim", type=int, default=16); p.add_argument("--region-seed", type=int, default=20261003); p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if len(args.reference_seeds) != 5 or len(args.model_seeds) != 5: raise ValueError("Exactly five reference seeds and five model seeds are required")
    main(args)
