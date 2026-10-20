"""Adaptive Real-Anchored TopK (ARA-TopK) for the frozen AG News protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "SyntheticData" / "exp"
VAL = EXP / "outputs" / "curation_validation"
CAR_CACHE = ROOT / "SyntheticData" / "CaRTry" / "outputs" / "car_selector_validation" / "cache" / "car_cluster_embeddings.pt"
OUT = ROOT / "SyntheticData" / "APT" / "outputs" / "ara_topk_validation"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}
sys.path.insert(0, str(EXP))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT)); p.add_argument("--validation-dir", default=str(VAL)); p.add_argument("--cluster-seeds", nargs="+", type=int, default=[41, 42]); p.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]); p.add_argument("--prepare-only", action="store_true"); p.add_argument("--evaluate-legacy400", action="store_true"); p.add_argument("--bert-model", default="bert-base-uncased"); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float); q = np.asarray(q, float); p = p / max(p.sum(), 1e-30); q = q / max(q.sum(), 1e-30); m = (p + q) / 2
    p_term = p[p > 0] * np.log(p[p > 0] / np.maximum(m[p > 0], 1e-30)); q_term = q[q > 0] * np.log(q[q > 0] / np.maximum(m[q > 0], 1e-30))
    return float(0.5 * p_term.sum() + 0.5 * q_term.sum())


def largest_remainder(raw: np.ndarray, total: int) -> np.ndarray:
    base = np.floor(raw).astype(int); left = int(total - base.sum()); frac = raw - base
    if left > 0:
        for i in np.argsort(-frac)[:left]: base[i] += 1
    elif left < 0:
        for i in np.argsort(frac)[: -left]:
            if base[i] > 0: base[i] -= 1
    return base


def load_data(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    data = curation.load_frozen_data(); scores = pd.read_csv(Path(args.validation_dir) / "sample_proximity_scores.csv"); topk = pd.read_csv(Path(args.validation_dir) / "e2_uniform_topk_fixed.csv")
    saved = torch.load(CAR_CACHE, weights_only=False); meta = saved.get("metadata", {})
    if meta.get("source") != "sentence-transformers" or int(meta.get("dimension", 0)) != 384: raise RuntimeError("MiniLM cache is missing or is not the required sentence-transformers embedding cache")
    emb = {k: np.asarray(v, dtype=float) for k, v in saved["embeddings"].items()}
    for k in emb: emb[k] /= np.maximum(np.linalg.norm(emb[k], axis=1, keepdims=True), 1e-12)
    return data, scores, topk, emb


def fit_real_clusters(real: np.ndarray, labels: np.ndarray, seed: int) -> dict[int, dict[str, Any]]:
    result = {}
    for label in LABEL_NAMES:
        idx = np.where(labels == label)[0]; k = min(max(2, round(math.sqrt(len(idx) / 2))), len(idx)); km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(real[idx]); local = km.labels_; counts = np.bincount(local, minlength=k).astype(float); result[label] = {"indices": idx, "k": k, "km": km, "real_dist": counts / counts.sum()}
    return result


def assign_clusters(emb: np.ndarray, clusters: dict[int, dict[str, Any]], labels: np.ndarray) -> np.ndarray:
    out = np.full(len(labels), -1, dtype=int)
    for label in LABEL_NAMES:
        idx = np.where(labels == label)[0]; km = clusters[label]["km"]; cent = km.cluster_centers_; cent = cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12); out[idx] = np.argmax(emb[idx] @ cent.T, axis=1)
    return out


def select_group(group: pd.DataFrame, quota: np.ndarray, cluster_ids: np.ndarray) -> tuple[set[str], dict[str, str]]:
    selected: set[str] = set(); reasons: dict[str, str] = {}; remaining = {}
    for cid, q in enumerate(quota):
        candidates = group[(cluster_ids == cid)].sort_values(["proximity_score", "sample_id"], ascending=[False, True])
        take = candidates.iloc[: min(int(q), len(candidates))]
        for sid in take.sample_id: selected.add(sid); reasons[sid] = "top_relevance_within_mode"
        remaining[cid] = candidates.iloc[min(int(q), len(candidates)):].copy()
    # Availability deficits are redistributed to the cluster furthest below P*.
    while len(selected) < 40:
        options = []
        counts = {cid: sum(cluster_ids[group.index.to_numpy() == i][0] == cid for i in group.index if group.loc[i, "sample_id"] in selected) for cid in range(len(quota))}
        for cid, rest in remaining.items():
            rest = rest[~rest.sample_id.isin(selected)]
            if len(rest): options.append((float(quota[cid] / 40 - counts[cid] / 40), float(rest.iloc[0].proximity_score), cid, rest.iloc[0]))
        if not options: break
        _, _, cid, row = max(options, key=lambda x: (x[0], x[1], -x[2])); selected.add(row.sample_id); reasons[row.sample_id] = "redistributed_quota"; remaining[cid] = remaining[cid][remaining[cid].sample_id != row.sample_id]
    if len(selected) != 40: raise RuntimeError(f"ARA quota could not fill 40 samples; got {len(selected)}")
    return selected, reasons


def prepare(args: argparse.Namespace) -> tuple[dict[int, list[dict[str, Any]]], pd.DataFrame]:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); data, scores, topk, emb = load_data(args); score_map = scores.set_index("sample_id").top5_real_cosine_mean.to_dict(); real_labels = np.asarray([x["label"] for x in data["real_reference"]]); real_emb = emb["real_reference"]; all_diagnostics = []; all_sets = {}
    candidate_lookup = {source: {x["sample_id"]: i for i, x in enumerate(data[f"{source}_train"])} for source in ("qwen", "llama")}
    for seed in args.cluster_seeds:
        clusters = fit_real_clusters(real_emb, real_labels, seed); selected_all = []
        for source in ("qwen", "llama"):
            train_key = f"{source}_train"; labels = np.asarray([x["label"] for x in data[train_key]]); assigned = assign_clusters(emb[train_key], clusters, labels)
            for label, cls in LABEL_NAMES.items():
                ids = [x["sample_id"] for x in data[train_key] if int(x["label"]) == label]; group = scores[scores.sample_id.isin(ids)].copy(); group["proximity_score"] = group.sample_id.map(score_map); group = group.sort_values(["proximity_score", "sample_id"], ascending=[False, True]); gidx = np.asarray([candidate_lookup[source][sid] for sid in group.sample_id]); cids = assigned[gidx]; info = clusters[label]; k = info["k"]; real_dist = info["real_dist"]; cand_counts = np.bincount(cids, minlength=k).astype(float); cand_dist = cand_counts / max(cand_counts.sum(), 1); top_ids = set(topk[(topk.source == source) & (topk.label == label)].sample_id); top_counts = np.bincount(np.asarray([cids[np.where(group.sample_id.to_numpy() == sid)[0][0]] for sid in top_ids]), minlength=k).astype(float); top_dist = top_counts / 40; alpha = jsd(top_dist, real_dist) / math.log(2); target = (1 - alpha) * top_dist + alpha * real_dist; quota = largest_remainder(40 * target, 40); chosen, reasons = select_group(group.reset_index(drop=True), quota, cids)
                selected_rows = group[group.sample_id.isin(chosen)].copy(); selected_clusters = cids[[sid in chosen for sid in group.sample_id]]; ara_counts = np.bincount(selected_clusters, minlength=k).astype(float); ara_dist = ara_counts / 40; top_frame = group[group.sample_id.isin(top_ids)]; ara_mean = selected_rows.proximity_score.mean(); top_mean = top_frame.proximity_score.mean(); changed = len(top_ids - chosen); jaccard = len(top_ids & chosen) / max(1, len(top_ids | chosen)); drop = top_mean - ara_mean; diag = {"ClusterSeed": seed, "Source": source, "Class": cls, "N_Real": len(info["indices"]), "N_Candidate": len(group), "K_Clusters": k, "Real_Entropy": float(-sum(x * math.log(x) for x in real_dist if x > 0)), "Candidate_Entropy": float(-sum(x * math.log(x) for x in cand_dist if x > 0)), "Candidate_JSD_to_Real": jsd(cand_dist, real_dist), "Candidate_Shift": jsd(cand_dist, real_dist) / math.log(2), "TopK_Entropy": float(-sum(x * math.log(x) for x in top_dist if x > 0)), "TopK_JSD_to_Real": jsd(top_dist, real_dist), "Alpha": alpha, "ARA_Entropy": float(-sum(x * math.log(x) for x in ara_dist if x > 0)), "ARA_JSD_to_Real": jsd(ara_dist, real_dist), "TopK_Cluster_Coverage": int((top_counts > 0).sum()) / k, "ARA_Cluster_Coverage": int((ara_counts > 0).sum()) / k, "TopK_Mean_Proximity": top_mean, "ARA_Mean_Proximity": ara_mean, "Proximity_Drop": drop, "Num_Changed_From_TopK": changed, "TopK_ARA_Jaccard": jaccard, "Warning": ";".join(x for x in ("LARGE_SELECTION_SHIFT" if changed > 20 or jaccard < .5 else "", "LARGE_RELEVANCE_DROP" if drop > .001 else "") if x)}; all_diagnostics.append(diag)
                for _, row in selected_rows.iterrows():
                    item = {"sample_id": row.sample_id, "source": row.source, "class": row["class"], "text": row.text, "label": int(row.label), "proximity_score": float(row.proximity_score), "real_cluster": int(cids[list(group.sample_id).index(row.sample_id)]), "cluster_quota": int(quota[cids[list(group.sample_id).index(row.sample_id)]]), "candidate_jsd": jsd(cand_dist, real_dist), "topk_jsd": jsd(top_dist, real_dist), "alpha": alpha, "selection_reason": reasons[row.sample_id]}; selected_all.append(item)
        selected_all.sort(key=lambda x: (int(x["label"]), x["sample_id"])); all_sets[seed] = selected_all; pd.DataFrame(selected_all).to_csv(out / f"ara_topk_cluster{seed}.csv", index=False)
    diag_frame = pd.DataFrame(all_diagnostics); diag_frame.to_csv(out / "ara_selection_diagnostics.csv", index=False); return all_sets, diag_frame


def train(args: argparse.Namespace, sets: dict[int, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    out = Path(args.output_dir); dev = pd.read_csv(Path(args.validation_dir) / "dev2000.csv").to_dict("records"); legacy = data["real_test"] if args.evaluate_legacy400 else None; train_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False); rows = []
    for cluster_seed, records in sets.items():
        for model_seed in args.model_seeds:
            run_name = f"ARA_TopK_cluster{cluster_seed}_model{model_seed}"; result = validation.train_one(records, dev, legacy, None, out, run_name, model_seed, train_args); rows.append(validation.metric_row("ARA-TopK", str(cluster_seed), model_seed, result, out / "training" / run_name))
    frame = pd.DataFrame(rows); frame.to_csv(out / "ara_run_results.csv", index=False); return frame


def main_table(args: argparse.Namespace, diag: pd.DataFrame, results: pd.DataFrame | None) -> None:
    out = Path(args.output_dir); top_results = pd.read_csv(Path(args.validation_dir) / "run_results.csv"); top_results = top_results[top_results.Experiment == "E2"].set_index("Model_Seed"); d = diag.groupby(["Source", "Class"]); rows = []
    for (source, cls), f in d:
        row = {"Source": source, "Class": cls}
        for col in ["Candidate_JSD", "TopK_JSD", "ARA_JSD", "Alpha", "TopK_Cluster_Coverage", "ARA_Cluster_Coverage", "TopK_Mean_Proximity", "ARA_Mean_Proximity", "Proximity_Drop", "Num_Changed_From_TopK", "TopK_ARA_Jaccard"]:
            source_col = {"Candidate_JSD": "Candidate_JSD_to_Real", "TopK_JSD": "TopK_JSD_to_Real", "ARA_JSD": "ARA_JSD_to_Real"}.get(col, col); row[col] = f[source_col].mean(); row[col + "_std"] = f[source_col].std(ddof=1)
        if results is not None:
            ara = results[results.Experiment == "ARA-TopK"]; top = top_results.loc[ara.Model_Seed.unique()]; row["TopK_Dev_F1"] = top.Dev_Macro_F1.mean(); row["ARA_Dev_F1"] = ara[(ara.Run_Dir.notna()) & (ara.Selection_Seed.astype(str).isin(["41", "42"]))].Dev_Macro_F1.mean(); row["ARA_minus_TopK_F1"] = row["ARA_Dev_F1"] - row["TopK_Dev_F1"]
        rows.append(row)
    main = pd.DataFrame(rows); perf = pd.DataFrame(); deltas = pd.DataFrame()
    if results is not None:
        ara = results[results.Experiment == "ARA-TopK"]; metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]; delta_rows = []
        for seed in args.model_seeds:
            a = ara[ara.Model_Seed == seed].groupby("Model_Seed")[metrics].mean().iloc[0]; t = top_results.loc[seed, metrics]; delta_rows.append({"Model_Seed": seed, **{m: float(a[m] - t[m]) for m in metrics}})
        dd = pd.DataFrame(delta_rows)
        summary_rows = [
            {"Model_Seed": "Mean", **{m: dd[m].mean() for m in metrics}},
            {"Model_Seed": "Std", **{m: dd[m].std(ddof=1) for m in metrics}},
            {"Model_Seed": "CI95_low", **{m: dd[m].mean() - 1.96 * dd[m].std(ddof=1) / math.sqrt(len(dd)) for m in metrics}},
            {"Model_Seed": "CI95_high", **{m: dd[m].mean() + 1.96 * dd[m].std(ddof=1) / math.sqrt(len(dd)) for m in metrics}},
        ]
        deltas = pd.concat([dd, pd.DataFrame(summary_rows)], ignore_index=True)
        perf = pd.DataFrame([{ "Selector": "TopK", **{m: top_results[m].mean() for m in metrics}}, {"Selector": "ARA-TopK cluster41", **{m: ara[ara.Selection_Seed == "41"][m].mean() for m in metrics}}, {"Selector": "ARA-TopK cluster42", **{m: ara[ara.Selection_Seed == "42"][m].mean() for m in metrics}}, {"Selector": "ARA-TopK Mean", **{m: ara[m].mean() for m in metrics}}])
    overall = "no advantage" if results is None else ("strong positive" if (deltas.loc[deltas.Model_Seed == "Mean", "Dev_Accuracy"].iloc[0] > 0 and (deltas.loc[deltas.Model_Seed.isin([42,43,44,45,46]), "Dev_Accuracy"] > 0).sum() == 5) else ("positive signal" if deltas.loc[deltas.Model_Seed == "Mean", "Dev_Accuracy"].iloc[0] > 0 and (deltas.loc[deltas.Model_Seed.isin([42,43,44,45,46]), "Dev_Accuracy"] > 0).sum() >= 4 else ("mixed" if (deltas.loc[deltas.Model_Seed.isin([42,43,44,45,46]), "Dev_Accuracy"] > 0).sum() == 3 else "no advantage")))
    with pd.ExcelWriter(out / "ARA_MAIN_TABLE.xlsx", engine="openpyxl") as w:
        main.to_excel(w, sheet_name="00_Main", index=False, startrow=0); start = len(main) + 3; pd.DataFrame({"Part": ["Selector performance summary"]}).to_excel(w, sheet_name="00_Main", index=False, startrow=start); perf.to_excel(w, sheet_name="00_Main", index=False, startrow=start + 1); start += len(perf) + 4; pd.DataFrame({"Conclusion": [overall]}).to_excel(w, sheet_name="00_Main", index=False, startrow=start); deltas.to_excel(w, sheet_name="00_Main", index=False, startrow=start + 2)
    status = overall if results is not None else "training pending"
    (out / "ara_analysis.txt").write_text(f"ARA-TopK conclusion: {status}\nCandidate/TopK/ARA diagnostics are in ara_selection_diagnostics.csv.\nFinal holdout was not evaluated.\n", encoding="utf-8")


def save_variance(args: argparse.Namespace, results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]
    rows = []
    ara = results[results.Experiment == "ARA-TopK"]
    for seed in args.model_seeds:
        a = ara[(ara.Model_Seed == seed) & (ara.Selection_Seed == "41")].iloc[0]
        b = ara[(ara.Model_Seed == seed) & (ara.Selection_Seed == "42")].iloc[0]
        rows.append({"Model_Seed": seed, **{m: abs(float(a[m]) - float(b[m])) for m in metrics}})
    frame = pd.DataFrame(rows); frame.to_csv(Path(args.output_dir) / "ara_cluster_variance.csv", index=False); return frame


def run(args: argparse.Namespace) -> None:
    if args.cluster_seeds != [41, 42] or args.model_seeds != [42, 43, 44, 45, 46]: raise ValueError("Protocol requires cluster seeds 41,42 and model seeds 42..46")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); data, _, _, _ = load_data(args); sets, diag = prepare(args)
    if args.prepare_only: main_table(args, diag, None); print(f"Prepared ARA-TopK selection under {out}"); return
    results = train(args, sets, data); save_variance(args, results); main_table(args, diag, results); print(results.to_string(index=False))


if __name__ == "__main__":
    run(args_parser())
