"""Real-Anchored Clustering and Ranking (RA-CaR) validation for AG News.

Ranking is exactly the frozen E2 Top-5 same-class real-proximity score.
Clustering is an independent semantic diagnostic/selection component; DSIR
scores and training loss weights are never used here.
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
import torch
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "SyntheticData" / "exp"
VALIDATION_DIR = EXP_DIR / "outputs" / "curation_validation"
LN_DIR = ROOT / "SyntheticData" / "DisrTry" / "outputs" / "ln_dsir_validation"
OUTPUT_DIR = ROOT / "SyntheticData" / "CaRTry" / "outputs" / "car_selector_validation"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

sys.path.insert(0, str(EXP_DIR))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--validation-dir", default=str(VALIDATION_DIR))
    p.add_argument("--ln-dir", default=str(LN_DIR))
    p.add_argument("--cluster-seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--n2", type=int, default=1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--cluster-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--evaluate-legacy400", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--merge-shards", action="store_true")
    p.add_argument("--bert-model", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    data = curation.load_frozen_data()
    dev = pd.read_csv(Path(args.validation_dir) / "dev2000.csv").to_dict("records")
    scores = pd.read_csv(Path(args.validation_dir) / "sample_proximity_scores.csv")
    topk = pd.read_csv(Path(args.validation_dir) / "e2_uniform_topk_fixed.csv")
    score_map = scores.set_index("sample_id")["top5_real_cosine_mean"]
    if not np.isfinite(score_map.to_numpy(dtype=float)).all():
        raise RuntimeError("E2 proximity cache contains non-finite scores")
    return data, dev, scores, topk


def length_ranking_diagnostics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    quartile_rows = []
    s = scores.copy()
    s["num_tokens"] = s.text.map(lambda x: len(str(x).split()))
    for (source, cls), frame in s.groupby(["source", "class"]):
        x = frame.num_tokens.to_numpy(float); y = frame.top5_real_cosine_mean.to_numpy(float)
        pear = float(pearsonr(x, y).statistic) if len(np.unique(x)) > 1 else np.nan
        spear = float(spearmanr(x, y).statistic) if len(np.unique(x)) > 1 else np.nan
        flag = "SEVERE_LENGTH_BIAS" if np.isfinite(spear) and abs(spear) > .7 else ("POSSIBLE_LENGTH_BIAS" if np.isfinite(spear) and abs(spear) > .5 else "")
        rows.append({"Source": source, "Class": cls, "Num_Candidates": len(frame), "Pearson_Length_Proximity": pear, "Spearman_Length_Proximity": spear, "Mean_Length": x.mean(), "Median_Length": np.median(x), "Mean_Proximity": y.mean(), "Std_Proximity": y.std(ddof=1), "Warning": flag})
        ranked = frame.sort_values(["top5_real_cosine_mean", "sample_id"], ascending=[False, True]).reset_index(drop=True)
        ranked["length_quartile"] = pd.qcut(ranked.num_tokens.rank(method="first"), 4, labels=["Q1_shortest", "Q2", "Q3", "Q4_longest"])
        selected = ranked.iloc[:40]
        counts = selected.length_quartile.value_counts().to_dict()
        cand_median = float(frame.num_tokens.median()); sel_median = float(selected.num_tokens.median())
        quartile_rows.append({"Source": source, "Class": cls, "Q1_count": int(counts.get("Q1_shortest", 0)), "Q2_count": int(counts.get("Q2", 0)), "Q3_count": int(counts.get("Q3", 0)), "Q4_count": int(counts.get("Q4_longest", 0)), "Selected_Median_Length": sel_median, "Candidate_Median_Length": cand_median, "Median_Length_Ratio": sel_median / max(cand_median, 1e-9), "Warning": "QUARTILE_CONCENTRATION" if max(counts.values()) > 28 else ""})
    return pd.DataFrame(rows), pd.DataFrame(quartile_rows)


def load_cluster_embeddings(data: dict[str, list[dict[str, Any]]], args: argparse.Namespace, out: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / "car_cluster_embeddings.pt"
    ids = [x["sample_id"] for key in ("qwen_train", "llama_train", "real_reference") for x in data[key]]
    if cache_path.exists():
        saved = torch.load(cache_path, weights_only=False)
        if saved.get("sample_ids") == ids:
            return {k: np.asarray(v) for k, v in saved["embeddings"].items()}, saved["metadata"]
    texts = [x["text"] for key in ("qwen_train", "llama_train", "real_reference") for x in data[key]]
    metadata = {"requested_model": args.cluster_embedding_model, "normalization": "L2", "source": "unknown"}
    arrays: dict[str, np.ndarray]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.cluster_embedding_model, local_files_only=True)
        vec = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        metadata.update({"source": "sentence-transformers", "model": args.cluster_embedding_model, "dimension": int(vec.shape[1])})
        arrays = {"qwen_train": vec[:320], "llama_train": vec[320:640], "real_reference": vec[640:]}
    except Exception as exc:
        # Existing E2 embeddings are a deterministic local fallback. The
        # manifest records this so the experiment never claims MiniLM was used.
        for source in ("qwen_train", "llama_train", "real_reference"):
            loaded = torch.load(Path(args.validation_dir) / "cache" / f"{source}_embeddings.pt", weights_only=False)
            arr = np.asarray(loaded["embeddings"], dtype=np.float32)
            arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
            arrays = locals().get("arrays", {}); arrays[source] = arr
        metadata.update({"source": "existing E2 RoBERTa embedding cache fallback", "dimension": 768, "fallback_reason": repr(exc)})
    torch.save({"sample_ids": ids, "embeddings": arrays, "metadata": metadata}, cache_path)
    return arrays, metadata


def select_group(frame: pd.DataFrame, emb: np.ndarray, seed: int, n2: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n = len(frame); k = min(max(2, round(math.sqrt(n / 2))), n)
    normalized = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    pca = PCA(n_components=0.95, random_state=seed)
    reduced = pca.fit_transform(normalized)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(reduced)
    distances = np.linalg.norm(reduced - km.cluster_centers_[labels], axis=1)
    f = frame.copy().reset_index(drop=True); f["cluster_id"] = labels; f["distance_to_centroid"] = distances
    rank_map = {idx: rank for rank, idx in enumerate(f.sort_values(["proximity_score", "sample_id"], ascending=[False, True]).index, start=1)}
    f["rank"] = f.index.map(rank_map).astype(int)
    core_budget = max(0, 40 - k * n2)
    order = f.sort_values(["proximity_score", "sample_id"], ascending=[False, True]).index.tolist()
    core = order[:core_budget]; selected = set(core); diversity = []
    for cid in range(k):
        candidates = [i for i in order if labels[i] == cid and i not in selected]
        for i in candidates[:n2]:
            if len(selected) < 40: selected.add(i); diversity.append(i)
    while len(selected) < 40:
        remaining = [i for i in order if i not in selected]
        if not remaining: break
        coverage = {cid: sum(labels[i] == cid for i in selected) / max(1, np.sum(labels == cid)) for cid in range(k)}
        i = min(remaining, key=lambda j: (coverage[labels[j]], -float(f.loc[j, "proximity_score"]), str(f.loc[j, "sample_id"])))
        selected.add(i); diversity.append(i)
    if len(selected) != 40: raise RuntimeError("RA-CaR could not fill fixed group budget")
    f["selected_core"] = f.index.isin(core); f["selected_diversity"] = f.index.isin(diversity); f["selected_car"] = f.index.isin(selected)
    diag = []
    for cid, g in f.groupby("cluster_id"):
        diag.append({"Cluster_ID": int(cid), "Cluster_Size": len(g), "Mean_Proximity": g.proximity_score.mean(), "Median_Proximity": g.proximity_score.median(), "Mean_Length": g.num_tokens.mean(), "Median_Length": g.num_tokens.median(), "Num_Core_Selected": int(g.selected_core.sum()), "Num_Diversity_Selected": int(g.selected_diversity.sum()), "Total_Selected": int(g.selected_car.sum())})
    med_lengths = f.groupby("cluster_id").num_tokens.median().to_numpy(float)
    try: kw = float(kruskal(*[g.num_tokens.to_numpy(float) for _, g in f.groupby("cluster_id")]).pvalue)
    except Exception: kw = np.nan
    meta = {"k_cluster": k, "original_dim": int(emb.shape[1]), "reduced_dim": int(reduced.shape[1]), "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()), "max_cluster_size_ratio": float(f.cluster_id.value_counts().max() / max(1, f.cluster_id.value_counts().min())), "cluster_median_length_ratio": float(med_lengths.max() / max(med_lengths.min(), 1e-9)), "length_centroid_corr": float(spearmanr(f.num_tokens, f.distance_to_centroid).statistic), "length_kruskal_p": kw}
    return f, pd.DataFrame(diag), meta


def prepare(args: argparse.Namespace) -> tuple[dict[int, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data, dev, scores, topk = load_inputs(args)
    score_map = scores.set_index("sample_id")["top5_real_cosine_mean"]
    scores["num_tokens"] = scores.text.map(lambda x: len(str(x).split()))
    emb, emb_meta = load_cluster_embeddings(data, args, out)
    length_bias, quartiles = length_ranking_diagnostics(scores)
    all_sets: dict[int, list[dict[str, Any]]] = {}; cluster_rows = []; quality_rows = []; coverage_rows = []; per_seed_frames = []
    for seed in args.cluster_seeds:
        selected_all = []
        for source in ("qwen", "llama"):
            for label, cls in LABEL_NAMES.items():
                records = [x for x in data[f"{source}_train"] if int(x["label"]) == label]
                ids = [x["sample_id"] for x in records]
                frame = scores.set_index("sample_id").loc[ids].reset_index()
                frame["proximity_score"] = [float(score_map[i]) for i in ids]; frame["num_tokens"] = [len(str(x["text"]).split()) for x in records]
                source_ids = [x["sample_id"] for x in data[f"{source}_train"]]
                source_indices = [source_ids.index(i) for i in ids]
                selected_frame, diag, meta = select_group(frame, emb[f"{source}_train"][source_indices], seed, args.n2)
                diag["Source"] = source; diag["Class"] = cls; diag["cluster_seed"] = seed; cluster_rows.extend(diag.to_dict("records"))
                for key in ("length_centroid_corr", "cluster_median_length_ratio", "length_kruskal_p"):
                    diag[key] = meta[key]
                diag["cluster_length_warning"] = np.where(diag.cluster_median_length_ratio > 2.5, "POSSIBLE_LENGTH_DRIVEN_CLUSTERING", "")
                cluster_rows[-len(diag):] = diag.to_dict("records")
                selected_ids = set(selected_frame.loc[selected_frame.selected_car, "sample_id"])
                core_ids = set(selected_frame.loc[selected_frame.selected_core, "sample_id"]); div_ids = set(selected_frame.loc[selected_frame.selected_diversity, "sample_id"])
                top_ids = set(topk[(topk.source == source) & (topk.label == label)].sample_id)
                car_scores = selected_frame[selected_frame.selected_car].proximity_score
                top_scores = frame.sort_values(["proximity_score", "sample_id"], ascending=[False, True]).iloc[:40].proximity_score
                selected_rank = selected_frame.loc[selected_frame.selected_car, "rank"]
                quality_rows.append({"Source": source, "Class": cls, "cluster_seed": seed, "TopK_mean_proximity": top_scores.mean(), "CaR_mean_proximity": car_scores.mean(), "quality_retention": car_scores.mean() / max(top_scores.mean(), 1e-12), "proximity_drop": top_scores.mean() - car_scores.mean(), "CaR_median_proximity": car_scores.median(), "CaR_p10_proximity": car_scores.quantile(.1), "CaR_p25_proximity": car_scores.quantile(.25), "CaR_mean_rank": selected_rank.mean(), "CaR_median_rank": selected_rank.median(), "CaR_max_rank": selected_rank.max(), "TopK_CaR_Jaccard": len(top_ids & selected_ids) / max(1, len(top_ids | selected_ids)), "num_removed_from_TopK": len(top_ids - selected_ids), "num_added_for_diversity": len(selected_ids - top_ids)})
                counts = selected_frame[selected_frame.selected_car].cluster_id.value_counts().reindex(range(meta["k_cluster"]), fill_value=0).to_numpy(float); probs = counts / 40; entropy = float(-sum(x * math.log(x) for x in probs if x > 0))
                top_cluster_counts = selected_frame[selected_frame.sample_id.isin(top_ids)].cluster_id.value_counts().reindex(range(meta["k_cluster"]), fill_value=0).to_numpy(float)
                coverage_rows.append({"Source": source, "Class": cls, "cluster_seed": seed, "num_clusters_covered": int((counts > 0).sum()), "total_clusters": meta["k_cluster"], "cluster_coverage_ratio": float((counts > 0).sum() / meta["k_cluster"]), "cluster_entropy": entropy, "TopK_num_clusters_covered": int((top_cluster_counts > 0).sum()), "TopK_cluster_coverage_ratio": float((top_cluster_counts > 0).sum() / meta["k_cluster"])})
                for i in selected_frame.index[selected_frame.selected_car]:
                    rec = dict(records[i]); rec["proximity_score"] = float(selected_frame.loc[i, "proximity_score"]); rec["selection_method"] = "RA-CaR"; rec.pop("original_index", None); selected_all.append(rec)
        selected_all.sort(key=lambda x: (int(x["label"]), x["sample_id"])); all_sets[seed] = selected_all; pd.DataFrame(selected_all).to_csv(out / f"car_sel{seed}.csv", index=False)
    # Add per-group cluster metadata to a compact artifact.
    cluster_diag = pd.DataFrame(cluster_rows); cluster_diag.to_csv(out / "car_cluster_diagnostics.csv", index=False)
    quality = pd.DataFrame(quality_rows); quality.to_csv(out / "car_quality_retention.csv", index=False)
    coverage = pd.DataFrame(coverage_rows); coverage.to_csv(out / "car_cluster_coverage.csv", index=False)
    length_bias.to_csv(out / "car_ranking_length_bias.csv", index=False); quartiles.to_csv(out / "car_length_quartile_selection.csv", index=False)
    (out / "car_protocol_manifest.json").write_text(json.dumps({"selector": "Real-Anchored Clustering and Ranking (RA-CaR)", "ranking_score": "frozen E2 top5_real_cosine_mean", "target": "QC train160 only for ranking", "allocation": {"qwen_per_class": 40, "llama_per_class": 40}, "cluster_formula": "round(sqrt(n/2))", "n2": args.n2, "cluster_seeds": args.cluster_seeds, "model_seeds": args.model_seeds, "embedding": emb_meta, "final_holdout_evaluated": False}, indent=2), encoding="utf-8")
    return all_sets, scores, topk, quality, coverage, cluster_diag, emb_meta


def diversity_table(data: dict[str, list[dict[str, Any]]], sets: dict[str, dict[int, list[dict[str, Any]]]], rank_embeddings: dict[str, Any], coverage: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    for source in ("qwen_train", "llama_train"):
        d = torch.load(Path(rank_embeddings[source]), weights_only=False) if isinstance(rank_embeddings[source], (str, Path)) else rank_embeddings[source]
        lookup.update(dict(zip(d["sample_ids"], np.asarray(d["embeddings"], dtype=float))))
    rows = []
    for method, by_seed in sets.items():
        for seed, records in by_seed.items():
            for source in ("qwen", "llama"):
                for label, cls in LABEL_NAMES.items():
                    group = [x for x in records if x["source"] == source and int(x["label"]) == label]; vec = np.asarray([lookup[x["sample_id"]] for x in group]); vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12); sim = vec @ vec.T; vals = sim[np.triu_indices(len(group), 1)]; nn = np.max(sim - np.eye(len(group)), axis=1); toks = [str(x["text"]).split() for x in group]; uni = [t for z in toks for t in z]; bi = [tuple(z[i:i+2]) for z in toks for i in range(len(z)-1)]
                    cov = coverage[(coverage.Source == source) & (coverage.Class == cls) & (coverage.cluster_seed == seed)]
                    rows.append({"Method": method, "selection_seed": seed, "Source": source, "Class": cls, "pairwise_cosine_mean": vals.mean(), "pairwise_cosine_median": np.median(vals), "pairwise_cosine_p90": np.quantile(vals, .9), "nearest_neighbor_mean": nn.mean(), "unique_unigram_ratio": len(set(uni)) / max(1, len(uni)), "unique_bigram_ratio": len(set(bi)) / max(1, len(bi)), "exact_duplicate_count": len(group) - len({x["text"] for x in group}), "cluster_coverage_ratio": float(cov.cluster_coverage_ratio.iloc[0]) if len(cov) else np.nan, "cluster_entropy": float(cov.cluster_entropy.iloc[0]) if len(cov) else np.nan})
    return pd.DataFrame(rows)


def train_car(sets: dict[int, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]], dev: list[dict[str, Any]], out: Path, args: argparse.Namespace) -> pd.DataFrame:
    legacy = data["real_test"] if args.evaluate_legacy400 else None; jobs = [(s, m, rows, f"RA_CaR_sel{s}_model{m}") for s, rows in sets.items() for m in args.model_seeds]
    if args.shard_index is not None: jobs = jobs[args.shard_index::args.num_shards]; train_dir = out / "shards" / f"shard_{args.shard_index}"
    else: train_dir = out
    train_dir.mkdir(parents=True, exist_ok=True); train_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False)
    rows = []
    for selection_seed, model_seed, records, run_name in jobs:
        result = validation.train_one(records, dev, legacy, None, train_dir, run_name, model_seed, train_args); rows.append(validation.metric_row("RA_CaR", str(selection_seed), model_seed, result, train_dir / "training" / run_name))
    frame = pd.DataFrame(rows); frame.to_csv(train_dir / "car_run_results.csv", index=False); return frame


def paired(combined: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]; rows = []
    for baseline in ("E1", "E2", "E6"):
        name = {"E1": "Random", "E2": "TopK", "E6": "LN-DSIR"}[baseline]
        for metric in metrics:
            vals = []
            for seed in sorted(combined.Model_Seed.unique()):
                car = combined[(combined.Experiment == "RA_CaR") & (combined.Model_Seed == seed)][metric].mean(); ref = combined[(combined.Experiment == baseline) & (combined.Model_Seed == seed)][metric]
                if len(ref): vals.append(float(car - (ref.mean() if baseline != "E2" else ref.iloc[0])))
            a = np.asarray(vals); margin = 1.96 * a.std(ddof=1) / math.sqrt(len(a)); rows.append({"Metric": metric, "Comparison": f"RA_CaR_minus_{name}", "Delta_Mean": a.mean(), "Delta_Std": a.std(ddof=1), "CI95_Low": a.mean() - margin, "CI95_High": a.mean() + margin, "Positive_Model_Seeds": int((a > 0).sum()), "N": len(a)})
    return pd.DataFrame(rows)


def merge(args: argparse.Namespace, prepared: tuple) -> None:
    sets, scores, topk, quality, coverage, cluster_diag, emb_meta = prepared; out = Path(args.output_dir); files = sorted((out / "shards").glob("shard_*/car_run_results.csv")); car = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    if len(car) != 25: raise RuntimeError(f"Expected 25 RA-CaR runs, found {len(car)}")
    baseline = pd.read_csv(Path(args.validation_dir) / "run_results.csv"); ln_path = Path(args.ln_dir) / "ln_dsir_run_results.csv"; ln = pd.read_csv(ln_path) if ln_path.exists() else pd.DataFrame(); combined = pd.concat([baseline, ln, car], ignore_index=True); combined.to_csv(out / "car_run_results.csv", index=False)
    specs = [("Random", "E1", baseline[baseline.Experiment == "E1"]), ("TopK", "E2", baseline[baseline.Experiment == "E2"]), ("Original DSIR", "E5", baseline[baseline.Experiment == "E5"]), ("LN-DSIR", "LN_DSIR", ln), ("RA-CaR", "RA_CaR", car)]; main_rows = []
    for name, exp, f in specs:
        row = {"Method": name, "Selector": exp, "Num_Selection_Sets": 1 if exp == "E2" else 5, "Num_Training_Runs": len(f)}
        for label, col in [("Accuracy", "Dev_Accuracy"), ("Macro_F1", "Dev_Macro_F1"), ("World_F1", "Dev_World_F1"), ("Sports_F1", "Dev_Sports_F1"), ("Business_F1", "Dev_Business_F1"), ("SciTech_F1", "Dev_SciTech_F1")]: row[f"{label}_Mean"] = f[col].mean() if len(f) else np.nan; row[f"{label}_Std"] = f[col].std(ddof=1) if len(f) > 1 else np.nan
        main_rows.append(row)
    main = pd.DataFrame(main_rows); pdelta = paired(combined); quality.to_csv(out / "car_quality_retention.csv", index=False); coverage.to_csv(out / "car_cluster_coverage.csv", index=False); pdelta.to_csv(out / "car_paired_deltas.csv", index=False)
    original_sets = {s: pd.read_csv(Path(ROOT / "SyntheticData/DisrTry/outputs/dsir_selector_validation") / f"e5_dsir_sel{s}.csv").to_dict("records") for s in args.cluster_seeds}; random_sets = {s: pd.read_csv(Path(args.validation_dir) / f"e1_random_sel{s}.csv").to_dict("records") for s in args.cluster_seeds}; all_sets = {"Random": random_sets, "TopK": {42: topk.to_dict("records")}, "LN-DSIR": {s: pd.read_csv(Path(args.ln_dir) / f"ln_dsir_sel{s}.csv").to_dict("records") for s in args.cluster_seeds}, "RA-CaR": sets}; rank_cache = {s: Path(args.validation_dir) / "cache" / f"{s}_embeddings.pt" for s in ("qwen_train", "llama_train")}; div = diversity_table(curation.load_frozen_data(), all_sets, rank_cache, coverage); div.to_csv(out / "car_diversity_comparison.csv", index=False)
    # Frequency and overlap artifacts.
    freq = Counter(x["sample_id"] for rows in sets.values() for x in rows); top_ids = set(topk.sample_id); fr = scores[["sample_id", "source", "class", "top5_real_cosine_mean"]].copy(); fr["car_selection_frequency"] = fr.sample_id.map(freq).fillna(0).astype(int); fr["topk_selected"] = fr.sample_id.isin(top_ids); fr.to_csv(out / "car_selection_frequency.csv", index=False)
    length_rows = []
    for method, by_seed in {"Random": random_sets, "TopK": {42: topk.to_dict("records")}, "RA-CaR": sets}.items():
        for seed, records in by_seed.items():
            for source in ("qwen", "llama"):
                for label, cls in LABEL_NAMES.items():
                    vals = np.asarray([len(str(x["text"]).split()) for x in records if x["source"] == source and int(x["label"]) == label], float); candidate_vals = scores[(scores.source == source) & (scores.label == label)].num_tokens.to_numpy(float); median_ratio = float(np.median(vals) / max(np.median(candidate_vals), 1e-9)); length_rows.append({"Method": method, "selection_seed": seed, "Source": source, "Class": cls, "mean_tokens": vals.mean(), "median_tokens": np.median(vals), "std_tokens": vals.std(ddof=1), "p10_tokens": np.quantile(vals, .1), "p90_tokens": np.quantile(vals, .9), "candidate_median_tokens": np.median(candidate_vals), "median_length_ratio_to_candidate": median_ratio, "Warning": "LENGTH_DISTRIBUTION_SHIFT" if method == "RA-CaR" and (median_ratio > 1.5 or median_ratio < .67) else ""})
    length = pd.DataFrame(length_rows); length.to_csv(out / "car_length_distribution.csv", index=False)
    overlap = quality[["Source", "Class", "cluster_seed", "TopK_CaR_Jaccard", "num_removed_from_TopK", "num_added_for_diversity"]]; overlap.to_csv(out / "car_topk_overlap.csv", index=False)
    with pd.ExcelWriter(out / "car_selector_results.xlsx", engine="openpyxl") as w:
        main.to_excel(w, sheet_name="Main_Comparison", index=False); main.to_excel(w, sheet_name="Per_Class_F1", index=False); pdelta[pdelta.Comparison.str.contains("TopK")].to_excel(w, sheet_name="TopK_vs_CaR_Delta", index=False); quality.to_excel(w, sheet_name="Quality_Retention", index=False); div.to_excel(w, sheet_name="Diversity", index=False); coverage.to_excel(w, sheet_name="Cluster_Coverage", index=False); overlap.to_excel(w, sheet_name="Overlap", index=False); length.to_excel(w, sheet_name="Length_Bias", index=False); cluster_diag.to_excel(w, sheet_name="Cluster_Diagnostics", index=False); fr.to_excel(w, sheet_name="Selection_Frequency", index=False)
    top_acc = pdelta[(pdelta.Comparison == "RA_CaR_minus_TopK") & (pdelta.Metric == "Dev_Accuracy")].iloc[0]; top_f1 = pdelta[(pdelta.Comparison == "RA_CaR_minus_TopK") & (pdelta.Metric == "Dev_Macro_F1")].iloc[0]; result = "strong improvement over TopK" if top_acc.Delta_Mean > 0 and top_f1.Delta_Mean > 0 else ("worse than TopK" if top_acc.Delta_Mean < 0 and top_f1.Delta_Mean < 0 else "small/mixed improvement")
    report = ["RA-CaR objective validation report", f"RA-CaR vs TopK Accuracy delta={top_acc.Delta_Mean:.6f}; positive model seeds={int(top_acc.Positive_Model_Seeds)}/5", f"RA-CaR vs TopK Macro-F1 delta={top_f1.Delta_Mean:.6f}; positive model seeds={int(top_f1.Positive_Model_Seeds)}/5", f"Mean quality retention={quality.quality_retention.mean():.6f}; mean proximity drop={quality.proximity_drop.mean():.6f}", f"Mean TopK/RA-CaR Jaccard={quality.TopK_CaR_Jaccard.mean():.6f}", f"Mean cluster coverage ratio={coverage.cluster_coverage_ratio.mean():.6f}", f"Ranking length Spearman median={pd.read_csv(out / 'car_ranking_length_bias.csv').Spearman_Length_Proximity.median():.6f}", f"Clustering embedding={emb_meta}", f"Conclusion: {result}", "Final holdout was not evaluated."]
    (out / "car_selector_analysis.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if len(args.cluster_seeds) != 5 or len(args.model_seeds) != 5: raise ValueError("Exactly five cluster and model seeds are required")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prepared = prepare(args)
    sets, _, _, _, _, _, _ = prepared
    if args.merge_shards:
        merge(args, prepared); return out
    if args.prepare_only:
        print(f"Prepared RA-CaR artifacts under {out}"); return out
    data, dev, _, _ = load_inputs(args); train_car(sets, data, dev, out, args); return out


if __name__ == "__main__":
    run(parse_args())
