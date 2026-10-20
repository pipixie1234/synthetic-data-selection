#!/usr/bin/env python3
"""Diagnose how much a balanced RealReference160 can vary from full AG News.

This script never trains a classifier. It samples 55 balanced references (40 per
class), embeds the full AG News train split once, and computes distributional
diagnostics for each reference plus the project's frozen current reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "SyntheticData"
PROTOCOL = SYN / "SmaExp" / "outputs" / "protocol"
SPLITS = SYN / "dimp" / "outputs" / "phase1" / "splits"
DEFAULT_OUT = SYN / "real_reference_uncertainty" / "outputs"
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DATASETS_CACHE = Path(os.environ.get("HF_DATASETS_CACHE", HF_HOME / "datasets"))
TRAIN_ARROW = DATASETS_CACHE.rglob("ag_news-train.arrow")
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}


def load_full_train() -> list[dict[str, Any]]:
    matches = list(TRAIN_ARROW)
    if not matches:
        raise FileNotFoundError("Cached ag_news-train.arrow was not found")
    ds = Dataset.from_file(str(matches[0]))
    return [{"sample_id": f"ag_train_{i}", "text": str(row["text"]), "label": int(row["label"]), "source_index": i}
            for i, row in enumerate(ds)]


def load_current_reference(full: list[dict[str, Any]]) -> dict[int, list[int]]:
    real_rows = [json.loads(x) for x in (PROTOCOL / "real_train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    qc_ids = json.loads((SPLITS / "real_qc_train_ids.json").read_text(encoding="utf-8"))
    ids = {int(real_rows[i]["source_index"]) for i in qc_ids}
    if len(ids) != 160:
        raise ValueError(f"Current real_qc_train reference has {len(ids)} unique source indices, expected 160")
    result = {label: [] for label in LABEL_NAMES}
    for row in full:
        if row["source_index"] in ids:
            result[row["label"]].append(row["source_index"])
    if any(len(v) != 40 for v in result.values()):
        raise ValueError(f"Current reference is not 40/class: { {LABEL_NAMES[k]: len(v) for k,v in result.items()} }")
    return result


def encode_texts(records: list[dict[str, Any]], model_name: str, batch_size: int, max_length: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            texts = [r["text"] for r in records[start:start + batch_size]]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state[:, 0, :].float().cpu().numpy()
            chunks.append(hidden)
            if (start // batch_size) % 25 == 0:
                print(f"Encoded {min(start + batch_size, len(records))}/{len(records)}")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def lengths(texts: list[str]) -> np.ndarray:
    # Keep the project's simple, model-independent token proxy reproducible.
    return np.asarray([len(str(x).split()) for x in texts], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def covariance_metrics(sample: np.ndarray, full_cov: np.ndarray, full_trace: float, full_diag: np.ndarray) -> dict[str, float]:
    cs = np.cov(sample, rowvar=False, ddof=1).astype(np.float64)
    diff = cs - full_cov
    cf_norm = float(np.linalg.norm(full_cov, ord="fro"))
    sample_trace = float(np.trace(cs))
    # Diagonal covariance correlation is more stable than a rank-deficient
    # determinant when n=40 and embedding dimension is 768.
    ds, df = np.diag(cs), full_diag
    diag_corr = float(np.corrcoef(ds, df)[0, 1]) if np.std(ds) > 0 and np.std(df) > 0 else 0.0
    return {
        "cov_fro_relative": float(np.linalg.norm(diff, ord="fro") / max(cf_norm, 1e-12)),
        "cov_trace_ratio": sample_trace / max(full_trace, 1e-12),
        "cov_diag_mean_ratio": float(ds.mean() / max(df.mean(), 1e-12)),
        "cov_diag_corr": diag_corr,
        "cov_effective_rank": float((np.trace(cs) ** 2) / max(np.trace(cs @ cs), 1e-12)),
    }


def median_bandwidth(full: np.ndarray, seed: int, pairs: int = 100_000) -> float:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(full), size=pairs)
    b = rng.integers(0, len(full), size=pairs)
    distances = np.linalg.norm(full[a] - full[b], axis=1)
    positive = distances[distances > 0]
    return float(max(np.median(positive), 1e-6))


def rbf_mean_cross(x: np.ndarray, y: np.ndarray, sigma: float, chunk: int = 2048) -> float:
    total = 0.0
    denom = 0
    for start in range(0, len(y), chunk):
        block = y[start:start + chunk]
        d2 = np.sum((x[:, None, :] - block[None, :, :]) ** 2, axis=2)
        total += float(np.exp(-d2 / (2 * sigma * sigma)).sum())
        denom += x.shape[0] * block.shape[0]
    return total / max(denom, 1)


def mmd_to_full(sample: np.ndarray, full_reference: np.ndarray, sigma: float) -> float:
    # Biased MMD^2. Compute the full-reference kernel in blocks: materializing
    # (reference_size, reference_size, embedding_dim) would otherwise exhaust
    # host RAM for reference sizes such as 4096.
    d2_x = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2)
    kxx = float(np.exp(-d2_x / (2 * sigma * sigma)).mean())
    kyy = rbf_mean_cross(full_reference, full_reference, sigma, chunk=256)
    kxy = rbf_mean_cross(sample, full_reference, sigma, chunk=256)
    return float(max(kxx + kyy - 2.0 * kxy, 0.0))


def prepare_clusters(full_emb: np.ndarray, labels: np.ndarray, pca_dim: int, k: int, seed: int) -> dict[int, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    # PCA is fitted on a balanced 2,000/class subset to make clustering cheap;
    # all full-class points are then assigned in the same representation.
    picked = []
    for label in LABEL_NAMES:
        idx = np.where(labels == label)[0]
        picked.extend(rng.choice(idx, size=min(2000, len(idx)), replace=False).tolist())
    pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")
    pca.fit(full_emb[picked])
    reduced = pca.transform(full_emb).astype(np.float32)
    result: dict[int, dict[str, Any]] = {}
    for label in LABEL_NAMES:
        idx = np.where(labels == label)[0]
        km = MiniBatchKMeans(n_clusters=k, random_state=seed + label, n_init=5, batch_size=2048, max_iter=100).fit(reduced[idx])
        assigned = km.labels_
        cent = km.cluster_centers_.astype(np.float32)
        top20: dict[int, set[int]] = {}
        for cid in range(k):
            local = np.where(assigned == cid)[0]
            distances = np.linalg.norm(reduced[idx[local]] - cent[cid], axis=1)
            order = local[np.argsort(distances)[:20]]
            top20[cid] = set(idx[order].tolist())
        all_distances = np.linalg.norm(reduced[idx][:, None, :] - cent[None, :, :], axis=2)
        result[label] = {"idx": idx, "reduced": reduced, "cluster_ids": assigned, "centers": cent,
                         "top20": top20, "min_full_centroid_distance": all_distances.min(axis=0)}
    return result


def cluster_metrics(selected: list[int], label: int, clusters: dict[int, dict[str, Any]]) -> dict[str, float]:
    info = clusters[label]
    idx = info["idx"]
    selected_set = set(selected)
    pos = {int(v): i for i, v in enumerate(idx)}
    selected_local = [pos[v] for v in selected if v in pos]
    assigned = info["cluster_ids"]
    occupied = len(set(assigned[selected_local])) if selected_local else 0
    centroid_hits = sum(bool(selected_set & info["top20"][cid]) for cid in range(len(info["centers"])))
    chosen = info["reduced"][selected]
    d = np.linalg.norm(chosen[:, None, :] - info["centers"][None, :, :], axis=2)
    min_selected = d.min(axis=0)
    min_full = info["min_full_centroid_distance"]
    return {
        "cluster_coverage_ratio": occupied / len(info["centers"]),
        "nearest_centroid_coverage": centroid_hits / len(info["centers"]),
        "nearest_centroid_distance_ratio": float(np.mean(min_selected / np.maximum(min_full, 1e-8))),
        "cluster_entropy": float(-sum(p * math.log(p) for p in np.bincount(assigned[selected_local], minlength=len(info["centers"])) / max(len(selected_local), 1) if p > 0)),
    }


def make_references(full: list[dict[str, Any]], current: dict[int, list[int]], seeds: list[int]) -> dict[str, dict[int, list[int]]]:
    by_class = {label: [r["source_index"] for r in full if r["label"] == label] for label in LABEL_NAMES}
    out: dict[str, dict[int, list[int]]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        out[f"seed_{seed}"] = {label: sorted(rng.choice(by_class[label], size=40, replace=False).tolist()) for label in LABEL_NAMES}
    out["current_reference"] = {label: sorted(v) for label, v in current.items()}
    return out


def summarize_metric(rows: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    random_rows = rows[rows.Reference.str.startswith("seed_")]
    out = []
    for metric in metric_cols:
        values = random_rows[metric].dropna().to_numpy(float)
        current = float(rows.loc[rows.Reference == "current_reference", metric].iloc[0])
        out.append({
            "Metric": metric,
            "Random55_Mean": float(values.mean()),
            "Random55_Std": float(values.std(ddof=1)),
            "Random55_P05": float(np.quantile(values, .05)),
            "Random55_Median": float(np.median(values)),
            "Random55_P95": float(np.quantile(values, .95)),
            "Current_Reference": current,
            "Current_Z_vs_Random55": float((current - values.mean()) / max(values.std(ddof=1), 1e-12)),
            "Current_Outside_P05_P95": bool(current < np.quantile(values, .05) or current > np.quantile(values, .95)),
        })
    return pd.DataFrame(out)


def pairwise_overlap(references: dict[str, dict[int, list[int]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [x for x in references if x.startswith("seed_")]
    rows = []
    per_seed = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            a_all = set(sum(references[a].values(), [])); b_all = set(sum(references[b].values(), []))
            inter = len(a_all & b_all); union = len(a_all | b_all)
            rows.append({"Reference_A": a, "Reference_B": b, "Intersection_Count": inter, "Jaccard": inter / max(union, 1)})
    pair = pd.DataFrame(rows)
    for name in names:
        vals = pair[(pair.Reference_A == name) | (pair.Reference_B == name)]
        per_seed.append({"Reference": name, "Mean_Overlap_Count": vals.Intersection_Count.mean(), "Mean_Jaccard": vals.Jaccard.mean()})
    return pair, pd.DataFrame(per_seed)


def run(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    full = load_full_train()
    labels = np.asarray([r["label"] for r in full], dtype=np.int64)
    texts = [r["text"] for r in full]
    current = load_current_reference(full)
    seeds = list(args.seeds)
    references = make_references(full, current, seeds)
    manifest = {"seeds": seeds, "per_class": 40, "current_reference": current,
                "full_count": len(full),
                "full_class_counts": {LABEL_NAMES[k]: int((labels == k).sum()) for k in LABEL_NAMES}}
    (out / "reference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Keep run outputs isolated while allowing the expensive 120k-row encoder
    # pass to be shared across configurations (for example, different MMD
    # reference sizes).
    cache_dir = Path(args.embedding_cache_dir).resolve() if args.embedding_cache_dir else out
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "ag_news_train_roberta_cls_embeddings.npy"
    meta_path = cache_dir / "ag_news_train_roberta_cls_embeddings.json"
    if emb_path.exists() and meta_path.exists():
        emb = np.load(emb_path, mmap_mode="r")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("count", -1)) != len(full) or int(meta.get("dim", -1)) != emb.shape[1]:
            raise RuntimeError("Existing embedding cache metadata does not match full AG News train")
        print(f"Reusing embedding cache: {emb_path}")
    else:
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        emb_arr = encode_texts(full, args.encoder_model, args.batch_size, args.max_length, device)
        np.save(emb_path, emb_arr)
        emb = np.load(emb_path, mmap_mode="r")
        meta = {"encoder_model": args.encoder_model, "pooling": "last_hidden_state[:,0,:]", "count": len(full), "dim": int(emb.shape[1]), "max_length": args.max_length}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        del emb_arr

    full_len = lengths(texts)
    print("Preparing full-data cluster representation...", flush=True)
    emb_arr = np.asarray(emb)
    clusters = prepare_clusters(emb_arr, labels, args.pca_dim, args.clusters, args.cluster_seed)
    full_kernel_refs = {}; sigmas = {}
    full_stats: dict[int, dict[str, Any]] = {}
    for label in LABEL_NAMES:
        idx = np.where(labels == label)[0]
        rng = np.random.default_rng(args.mmd_seed + label)
        ref_idx = rng.choice(idx, size=min(args.mmd_reference_size, len(idx)), replace=False)
        full_kernel_refs[label] = emb_arr[ref_idx].astype(np.float32)
        sigmas[label] = median_bandwidth(emb_arr[idx], args.mmd_seed + label, args.bandwidth_pairs)
        print(f"Precomputing full statistics for {LABEL_NAMES[label]}...", flush=True)
        cov = np.cov(emb_arr[idx], rowvar=False, ddof=1).astype(np.float64)
        full_stats[label] = {"mean": emb_arr[idx].mean(axis=0), "cov": cov,
                             "trace": float(np.trace(cov)), "diag": np.diag(cov),
                             "length_mean": float(full_len[idx].mean()), "length_std": float(full_len[idx].std(ddof=1))}

    metric_rows = []
    partial_path = out / "reference_class_metrics_partial.csv"
    if partial_path.exists():
        try:
            previous = pd.read_csv(partial_path)
            if set(previous.columns) >= {"Reference", "Class"}:
                metric_rows = previous.to_dict("records")
                print(f"Resuming from {partial_path}: {previous.Reference.nunique()} references already complete", flush=True)
        except Exception as exc:
            print(f"Ignoring invalid partial metrics file: {exc}", flush=True)
    completed = {str(row["Reference"]) for row in metric_rows}
    for ref_name, by_class in references.items():
        if ref_name in completed:
            continue
        all_ids = sum(by_class.values(), [])
        for label in LABEL_NAMES:
            selected = by_class[label]
            idx = np.where(labels == label)[0]
            sample_emb = np.asarray(emb)[selected]
            full_emb = emb_arr[idx]
            sample_len = full_len[selected]; base_len = full_len[idx]
            mean_s = sample_emb.mean(axis=0); mean_f = full_emb.mean(axis=0)
            row = {"Reference": ref_name, "Class": LABEL_NAMES[label], "Label": label, "N": len(selected),
                   "Length_Mean": float(sample_len.mean()), "Length_Std": float(sample_len.std(ddof=1)), "Length_Median": float(np.median(sample_len)),
                   "Length_P10": float(np.quantile(sample_len, .1)), "Length_P90": float(np.quantile(sample_len, .9)),
                   "Full_Length_Mean": full_stats[label]["length_mean"], "Length_Mean_Delta": float(sample_len.mean() - full_stats[label]["length_mean"]),
                   "Length_Mean_Z": float((sample_len.mean() - full_stats[label]["length_mean"]) / max(full_stats[label]["length_std"], 1e-12)),
                   "Embedding_Mean_L2": float(np.linalg.norm(mean_s - full_stats[label]["mean"])), "Embedding_Mean_Relative_L2": float(np.linalg.norm(mean_s - full_stats[label]["mean"]) / max(np.linalg.norm(full_stats[label]["mean"]), 1e-12)),
                   "Embedding_Mean_Cosine": cosine(mean_s, full_stats[label]["mean"]),
                   "MMD2_to_Full": mmd_to_full(sample_emb, full_kernel_refs[label], sigmas[label]),
                   "MMD_Bandwidth": sigmas[label]}
            row.update(covariance_metrics(sample_emb, full_stats[label]["cov"], full_stats[label]["trace"], full_stats[label]["diag"]))
            row.update(cluster_metrics(selected, label, clusters))
            metric_rows.append(row)
        print(f"Finished {ref_name}", flush=True)
        pd.DataFrame(metric_rows).to_csv(partial_path, index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "reference_class_metrics.csv", index=False)

    pair, pair_seed = pairwise_overlap(references)
    pair.to_csv(out / "pairwise_overlap.csv", index=False); pair_seed.to_csv(out / "pairwise_overlap_per_seed.csv", index=False)
    metric_names = ["Length_Mean_Z", "Embedding_Mean_Relative_L2", "Embedding_Mean_Cosine", "MMD2_to_Full", "cov_fro_relative", "cov_trace_ratio", "cluster_coverage_ratio", "nearest_centroid_coverage", "nearest_centroid_distance_ratio"]
    summary_rows = []
    for label_name in list(LABEL_NAMES.values()) + ["Overall"]:
        subset = metrics if label_name == "Overall" else metrics[metrics.Class == label_name]
        # Overall is a macro-average over classes for each reference.
        macro = subset.groupby("Reference", as_index=False)[metric_names].mean() if label_name == "Overall" else subset.copy()
        summary_rows.append(summarize_metric(macro, metric_names).assign(Class=label_name))
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(out / "reference_uncertainty_summary.csv", index=False)

    current_flags = summary[(summary.Class == "Overall") & summary.Current_Outside_P05_P95]
    decision = "real-reference uncertainty is a plausible method component" if len(current_flags) >= args.flag_metric_count else "real sampling variation appears small; continue targeted generation"
    report = {
        "decision": decision,
        "decision_rule": f"Flag current reference if at least {args.flag_metric_count} overall diagnostics are outside the 5th-95th percentile of 55 random references",
        "flagged_overall_metrics": current_flags.Metric.tolist(),
        "random_reference_count": len(seeds), "per_class_n": 40, "full_train_count": len(full),
        "embedding": meta, "mmd": {"estimator": "biased MMD^2", "full_kernel_reference_size": args.mmd_reference_size, "bandwidth": "median random full-class pair distance", "note": "full AG News train is represented by a deterministic large subset for both kernel terms; this is a computational approximation to full-train MMD"},
        "pairwise_overlap": {"mean_intersection": float(pair.Intersection_Count.mean()), "mean_jaccard": float(pair.Jaccard.mean())},
    }
    (out / "diagnostic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(out / "real_reference_uncertainty.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Class_Metrics", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        pair.to_excel(writer, sheet_name="Pairwise_Overlap", index=False)
        pair_seed.to_excel(writer, sheet_name="Overlap_Per_Seed", index=False)
        pd.DataFrame([report]).to_excel(writer, sheet_name="Decision", index=False)
        for ws in writer.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(max(len(str(x.value or "")) for x in col) + 2, 32)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out / 'real_reference_uncertainty.xlsx'}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--embedding-cache-dir", default="", help="Optional directory containing/saving the full-train embedding cache")
    p.add_argument("--encoder-model", default=str(ROOT / "cloud" / "model" / "roberta-base"))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 56)))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="")
    p.add_argument("--pca-dim", type=int, default=16)
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--cluster-seed", type=int, default=20261001)
    p.add_argument("--mmd-seed", type=int, default=20261002)
    p.add_argument("--mmd-reference-size", type=int, default=1024)
    p.add_argument("--bandwidth-pairs", type=int, default=20000)
    p.add_argument("--flag-metric-count", type=int, default=2)
    args = p.parse_args()
    if len(args.seeds) != 55 or len(set(args.seeds)) != 55:
        raise ValueError("Exactly 55 unique random reference seeds are required")
    run(args)


if __name__ == "__main__":
    main()
