"""Distribution-Guided Synthetic Data Curation pilot.

This pilot deliberately uses only the frozen Phase 1 train pools for curation.
The real test split is passed to the CE trainer only for final evaluation.
"""
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
from scipy.spatial.distance import cdist
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "SyntheticData"
PROTOCOL_DIR = SYNTHETIC_DIR / "SmaExp" / "outputs" / "protocol"
SPLITS_DIR = SYNTHETIC_DIR / "dimp" / "outputs" / "phase1" / "splits"
DEFAULT_OUTPUT = SYNTHETIC_DIR / "exp" / "outputs" / "distribution_curation" / "pilot"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}
LABEL_DISPLAY = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_frozen_data() -> dict[str, list[dict[str, Any]]]:
    """Load the exact frozen Phase 1 train pools and real test set."""
    qwen = load_jsonl(PROTOCOL_DIR / "synthetic_qwen.jsonl")
    llama = load_jsonl(PROTOCOL_DIR / "synthetic_llama.jsonl")
    real_train = load_jsonl(PROTOCOL_DIR / "real_train.jsonl")
    real_test = load_jsonl(PROTOCOL_DIR / "real_test.jsonl")
    qwen_train_ids = load_json(SPLITS_DIR / "qwen_train_ids.json")
    llama_train_ids = load_json(SPLITS_DIR / "llama_train_ids.json")
    qc_train_ids = load_json(SPLITS_DIR / "real_qc_train_ids.json")

    def records(rows: list[dict[str, Any]], ids: list[int], source: str) -> list[dict[str, Any]]:
        result = []
        for idx in ids:
            row = rows[idx]
            result.append({
                "sample_id": f"{source}_{idx}",
                "source": source,
                "label": int(row["label"]),
                "text": str(row["text"]),
                "original_index": idx,
            })
        return result

    data = {
        "qwen_train": records(qwen, qwen_train_ids, "qwen"),
        "llama_train": records(llama, llama_train_ids, "llama"),
        "real_reference": records(real_train, qc_train_ids, "real"),
        "real_test": [
            {"sample_id": f"real_test_{i}", "source": "real_test", "label": int(row["label"]), "text": str(row["text"])}
            for i, row in enumerate(real_test)
        ],
    }
    return data


def check_frozen_inputs(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["qwen_train_count"] = len(data["qwen_train"])
    checks["llama_train_count"] = len(data["llama_train"])
    checks["real_reference_count"] = len(data["real_reference"])
    checks["real_test_count"] = len(data["real_test"])
    checks["synthetic_train_only"] = all(x["original_index"] not in set(load_json(SPLITS_DIR / "qwen_test_ids.json")) for x in data["qwen_train"]) and all(x["original_index"] not in set(load_json(SPLITS_DIR / "llama_test_ids.json")) for x in data["llama_train"])
    checks["reference_not_test"] = not ({x["text"] for x in data["real_reference"]} & {x["text"] for x in data["real_test"]})
    checks["all_train_ids_unique"] = len({x["sample_id"] for key in ("qwen_train", "llama_train") for x in data[key]}) == 640
    checks["class_counts"] = {
        key: {LABEL_NAMES[label]: sum(x["label"] == label for x in data[key]) for label in LABEL_NAMES}
        for key in ("qwen_train", "llama_train", "real_reference", "real_test")
    }
    checks["passed"] = bool(checks["synthetic_train_only"] and checks["reference_not_test"] and checks["all_train_ids_unique"] and checks["qwen_train_count"] == 320 and checks["llama_train_count"] == 320 and checks["real_reference_count"] == 160 and checks["real_test_count"] == 400)
    if not checks["passed"]:
        raise RuntimeError(f"Frozen-input sanity check failed: {checks}")
    return checks


def embedding_cache_path(output_dir: Path, source: str) -> Path:
    names = {"real_reference": "real_reference_embeddings.pt", "qwen_train": "qwen_train_embeddings.pt", "llama_train": "llama_train_embeddings.pt"}
    return output_dir / "cache" / names[source]


def compute_embeddings(records: list[dict[str, Any]], model_name: str, batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            texts = [r["text"] for r in records[start:start + batch_size]]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            batches.append(model(**encoded).last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def load_or_compute_embeddings(data: dict[str, list[dict[str, Any]]], output_dir: Path, model_name: str, batch_size: int) -> dict[str, dict[str, Any]]:
    result = {}
    for source in ("real_reference", "qwen_train", "llama_train"):
        path = embedding_cache_path(output_dir, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_ids = [r["sample_id"] for r in data[source]]
        loaded = None
        if path.exists():
            loaded = torch.load(path, weights_only=False)
            valid = loaded.get("metadata", {}).get("encoder_name") == model_name and loaded.get("metadata", {}).get("pooling") == "cls_first_token" and loaded.get("sample_ids") == expected_ids
            if not valid:
                loaded = None
        if loaded is None:
            print(f"Computing {source} embeddings with {model_name}...")
            matrix = compute_embeddings(data[source], model_name, batch_size)
            loaded = {
                "sample_ids": expected_ids,
                "source": source,
                "labels": [r["label"] for r in data[source]],
                "embeddings": matrix,
                "metadata": {"encoder_name": model_name, "pooling": "cls_first_token", "embedding_dim": int(matrix.shape[1])},
            }
            torch.save(loaded, path)
        result[source] = loaded
    return result


def median_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    distances = cdist(x, y, metric="euclidean")
    sigma = float(np.median(distances))
    return max(sigma, 1e-12)


def unbiased_mmd(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    kxx = np.exp(-cdist(x, x, metric="sqeuclidean") / (2 * sigma * sigma))
    kyy = np.exp(-cdist(y, y, metric="sqeuclidean") / (2 * sigma * sigma))
    kxy = np.exp(-cdist(x, y, metric="sqeuclidean") / (2 * sigma * sigma))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    n = len(x)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean())


def compute_class_mmd(real: np.ndarray, synthetic: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    n = min(len(real), len(synthetic))
    if n < 2:
        raise ValueError("MMD requires at least two samples per class")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        real_sample = real[rng.choice(len(real), size=n, replace=False)]
        synthetic_sample = synthetic[rng.choice(len(synthetic), size=n, replace=False)]
        values.append(unbiased_mmd(synthetic_sample, real_sample, median_bandwidth(synthetic_sample, real_sample)))
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std()), "sample_size": int(n), "repeats": int(repeats)}


def compute_soft_allocation(embeddings: dict[str, dict[str, Any]], budget: int, repeats: int, seed: int) -> tuple[pd.DataFrame, float]:
    rows = []
    for label, class_name in LABEL_NAMES.items():
        real_idx = np.asarray(embeddings["real_reference"]["labels"]) == label
        qwen_idx = np.asarray(embeddings["qwen_train"]["labels"]) == label
        llama_idx = np.asarray(embeddings["llama_train"]["labels"]) == label
        qwen_stats = compute_class_mmd(embeddings["real_reference"]["embeddings"][real_idx], embeddings["qwen_train"]["embeddings"][qwen_idx], repeats, seed + label * 1009 + 1)
        llama_stats = compute_class_mmd(embeddings["real_reference"]["embeddings"][real_idx], embeddings["llama_train"]["embeddings"][llama_idx], repeats, seed + label * 1009 + 2)
        rows.append({"Class": class_name, "label": label, "Qwen_MMD_mean": qwen_stats["mean"], "Qwen_MMD_std": qwen_stats["std"], "Llama_MMD_mean": llama_stats["mean"], "Llama_MMD_std": llama_stats["std"], "MMD_Difference": abs(qwen_stats["mean"] - llama_stats["mean"]), "sample_size": qwen_stats["sample_size"], "repeats": repeats})
    tau = max(float(np.median([row["MMD_Difference"] for row in rows])), 1e-6)
    for row in rows:
        q = math.exp(-row["Qwen_MMD_mean"] / tau)
        l = math.exp(-row["Llama_MMD_mean"] / tau)
        row["Tau"] = tau
        row["Qwen_Ratio"] = q / (q + l)
        row["Llama_Ratio"] = 1.0 - row["Qwen_Ratio"]
        q_count = int(round(budget * row["Qwen_Ratio"]))
        q_available = int(np.sum(np.asarray(embeddings["qwen_train"]["labels"]) == row["label"]))
        l_available = int(np.sum(np.asarray(embeddings["llama_train"]["labels"]) == row["label"]))
        q_count = min(max(q_count, 0), q_available)
        l_count = budget - q_count
        if l_count > l_available:
            l_count = l_available
            q_count = budget - l_count
        if q_count + l_count != budget or q_count > q_available or l_count > l_available:
            raise RuntimeError(f"Availability constraint cannot satisfy budget for {row['Class']}")
        row["Qwen_Count"] = q_count
        row["Llama_Count"] = l_count
    return pd.DataFrame(rows), tau


def proximity_scores(data: dict[str, list[dict[str, Any]]], embeddings: dict[str, dict[str, Any]], top_k: int) -> pd.DataFrame:
    real = embeddings["real_reference"]
    real_by_label = {label: real["embeddings"][np.asarray(real["labels"]) == label] for label in LABEL_NAMES}
    rows = []
    for source in ("qwen_train", "llama_train"):
        emb = embeddings[source]
        for i, record in enumerate(data[source]):
            same = real_by_label[record["label"]]
            sample = emb["embeddings"][i:i + 1]
            sample = sample / np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12)
            real_norm = same / np.maximum(np.linalg.norm(same, axis=1, keepdims=True), 1e-12)
            sims = np.sort((sample @ real_norm.T).ravel())[::-1]
            k = min(top_k, len(sims))
            chosen = sims[:k]
            rows.append({"sample_id": record["sample_id"], "source": record["source"], "class": LABEL_NAMES[record["label"]], "label": record["label"], "text": record["text"], "top5_real_cosine_mean": float(chosen.mean()), "top1_real_cosine": float(chosen[0]), "top5_real_cosine_std": float(chosen.std()), "selected_E2": False, "selected_E4": False})
    scores = pd.DataFrame(rows)
    if not np.isfinite(scores[["top5_real_cosine_mean", "top1_real_cosine", "top5_real_cosine_std"]].to_numpy()).all():
        raise RuntimeError("Proximity scores contain NaN/Inf")
    return scores


def group_records(data: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups = defaultdict(list)
    for row in data:
        groups[int(row["label"])].append(row)
    return {label: sorted(groups[label], key=lambda x: x["sample_id"]) for label in LABEL_NAMES}


def select_dataset(data: dict[str, list[dict[str, Any]]], scores: pd.DataFrame, allocation: pd.DataFrame, method: str, budget: int, selection_seed: int) -> list[dict[str, Any]]:
    score_map = scores.set_index("sample_id")
    alloc_map = {int(row.label): (int(row.Qwen_Count), int(row.Llama_Count)) for row in allocation.itertuples()}
    selected = []
    for label in LABEL_NAMES:
        for source, source_idx in (("qwen", 0), ("llama", 1)):
            pool = group_records(data[f"{source}_train"])[label]
            count = budget // 2 if method.startswith("uniform") else alloc_map[label][source_idx]
            if "topk" in method:
                pool = sorted(pool, key=lambda x: (-float(score_map.loc[x["sample_id"], "top5_real_cosine_mean"]), x["sample_id"]))[:count]
            else:
                rng = random.Random(selection_seed + label * 1009 + source_idx)
                pool = rng.sample(pool, count)
            selected.extend(pool)
    output = []
    for row in sorted(selected, key=lambda x: (int(x["label"]), x["sample_id"])):
        item = dict(row)
        item["proximity_score"] = float(score_map.loc[item["sample_id"], "top5_real_cosine_mean"])
        item["selection_method"] = method
        item.pop("original_index", None)
        output.append(item)
    return output


def mark_selected(scores: pd.DataFrame, selected_e2: list[dict[str, Any]], selected_e4: list[dict[str, Any]]) -> pd.DataFrame:
    e2_ids = {x["sample_id"] for x in selected_e2}
    e4_ids = {x["sample_id"] for x in selected_e4}
    scores = scores.copy()
    scores["selected_E2"] = scores["sample_id"].isin(e2_ids)
    scores["selected_E4"] = scores["sample_id"].isin(e4_ids)
    return scores


def duplicate_stats(records: list[dict[str, Any]], embedding_lookup: dict[str, np.ndarray]) -> dict[str, Any]:
    texts = [x["text"] for x in records]
    exact = len(texts) - len(set(texts))
    vectors = np.asarray([embedding_lookup[x["sample_id"]] for x in records])
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sims = vectors @ vectors.T
    # CLS representations are globally high-cosine for this encoder; 0.999 is
    # a conservative warning threshold and does not affect selection.
    near_pairs = int(np.sum(np.triu(sims >= 0.999, k=1)))
    return {"exact_duplicate_count": exact, "near_duplicate_threshold": 0.999, "near_duplicate_pairs_cosine_ge_0_999": near_pairs, "near_duplicate_warning": near_pairs > 0}


def train_experiments(datasets: dict[str, list[dict[str, Any]]], output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    # Import after paths are set; this keeps the pilot tied to the existing CE trainer.
    sys.path.insert(0, str(SYNTHETIC_DIR / "dimp"))
    from phase1_trainers import Phase1Trainer

    trainer = Phase1Trainer(model_name=args.bert_model, num_labels=4, loss_type="ce")
    results = []
    names = [("E1", "uniform_random"), ("E2", "uniform_topk"), ("E3", "soft_random"), ("E4", "soft_topk")]
    for experiment, method in names:
        result = trainer.train(datasets[method], datasets["real_test"], quality_scores=None, output_dir=output_dir / "training", run_name=experiment, seed=args.model_seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, save_checkpoints=False)
        metrics = result["metrics"]
        per_class = result["per_class_metrics"]
        results.append({"Experiment": experiment, "Allocation": "Uniform" if method.startswith("uniform") else "MMD Soft", "Selection": "Random" if method.endswith("random") else "TopK", "Train_Count": len(datasets[method]), "Accuracy": metrics["eval_accuracy"], "Macro_F1": metrics["eval_macro_f1"], "World_F1": per_class.get("World", {}).get("f1-score", np.nan), "Sports_F1": per_class.get("Sports", {}).get("f1-score", np.nan), "Business_F1": per_class.get("Business", {}).get("f1-score", np.nan), "SciTech_F1": per_class.get("Sci/Tech", per_class.get("SciTech", {})).get("f1-score", np.nan)})
    return results


def save_excel(output_dir: Path, results: pd.DataFrame, allocation: pd.DataFrame, scores: pd.DataFrame, datasets: dict[str, list[dict[str, Any]]]) -> None:
    selected_summary = []
    for method in ("uniform_random", "uniform_topk", "soft_random", "soft_topk"):
        frame = pd.DataFrame(datasets[method])
        counts = frame.groupby(["source", "label"]).size().reset_index(name="count")
        counts["Experiment"] = method
        selected_summary.append(counts)
    proximity_stats = scores.groupby(["source", "class"])["top5_real_cosine_mean"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    with pd.ExcelWriter(output_dir / "distribution_curation_pilot.xlsx", engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Pilot_Results", index=False)
        allocation.to_excel(writer, sheet_name="Allocation", index=False)
        results[["Experiment", "World_F1", "Sports_F1", "Business_F1", "SciTech_F1"]].to_excel(writer, sheet_name="Per_Class_F1", index=False)
        pd.concat(selected_summary, ignore_index=True).to_excel(writer, sheet_name="Selected_Data_Summary", index=False)
        proximity_stats.to_excel(writer, sheet_name="Proximity_Stats", index=False)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.model_seed)
    data = load_frozen_data()
    validation = check_frozen_inputs(data)
    embeddings = load_or_compute_embeddings(data, output_dir, args.encoder_model, args.embedding_batch_size)
    allocation, tau = compute_soft_allocation(embeddings, args.per_class_budget, args.mmd_repeats, args.selection_seed)
    allocation.to_csv(output_dir / "allocation.csv", index=False)
    print(allocation[["Class", "Qwen_MMD_mean", "Qwen_MMD_std", "Llama_MMD_mean", "Llama_MMD_std", "MMD_Difference", "Tau", "Qwen_Ratio", "Llama_Ratio", "Qwen_Count", "Llama_Count"]].to_string(index=False))
    print(f"final tau={tau:.8f}")

    scores = proximity_scores(data, embeddings, args.top_k)
    datasets = {
        "uniform_random": select_dataset(data, scores, allocation, "uniform_random", args.per_class_budget, args.selection_seed),
        "uniform_topk": select_dataset(data, scores, allocation, "uniform_topk", args.per_class_budget, args.selection_seed),
        "soft_random": select_dataset(data, scores, allocation, "soft_random", args.per_class_budget, args.selection_seed),
        "soft_topk": select_dataset(data, scores, allocation, "soft_topk", args.per_class_budget, args.selection_seed),
        "real_test": data["real_test"],
    }
    scores = mark_selected(scores, datasets["uniform_topk"], datasets["soft_topk"])
    scores.to_csv(output_dir / "sample_proximity_scores.csv", index=False)
    for method, filename in (("uniform_random", "e1_uniform_random.csv"), ("uniform_topk", "e2_uniform_topk.csv"), ("soft_random", "e3_soft_random.csv"), ("soft_topk", "e4_soft_topk.csv")):
        pd.DataFrame(datasets[method]).to_csv(output_dir / filename, index=False)

    embedding_lookup = {}
    for source in ("qwen_train", "llama_train"):
        embedding_lookup.update(dict(zip(embeddings[source]["sample_ids"], embeddings[source]["embeddings"])))
    validation["experiments"] = {}
    for method in ("uniform_random", "uniform_topk", "soft_random", "soft_topk"):
        frame = pd.DataFrame(datasets[method])
        counts = Counter(frame["label"])
        validation["experiments"][method] = {"train_count": len(frame), "class_counts": {LABEL_NAMES[k]: counts[k] for k in LABEL_NAMES}, "sample_ids_unique": frame["sample_id"].is_unique, "duplicate_stats": duplicate_stats(datasets[method], embedding_lookup)}
        if len(frame) != args.per_class_budget * 4 or any(counts[k] != args.per_class_budget for k in LABEL_NAMES):
            raise RuntimeError(f"Dataset sanity check failed for {method}")
        if method.startswith("uniform"):
            source_counts = frame.groupby(["label", "source"]).size().to_dict()
            if any(source_counts.get((k, source), 0) != args.per_class_budget // 2 for k in LABEL_NAMES for source in ("qwen", "llama")):
                raise RuntimeError(f"Uniform source balance failed for {method}")
        else:
            for row in allocation.itertuples():
                frame_row = frame[frame["label"] == row.label]
                if int((frame_row["source"] == "qwen").sum()) != int(row.Qwen_Count) or int((frame_row["source"] == "llama").sum()) != int(row.Llama_Count):
                    raise RuntimeError(f"Allocation count failed for {method}/{row.Class}")
    validation["mmd_finite"] = bool(np.isfinite(allocation[["Qwen_MMD_mean", "Qwen_MMD_std", "Llama_MMD_mean", "Llama_MMD_std"]].to_numpy()).all())
    validation["proximity_finite"] = bool(np.isfinite(scores[["top5_real_cosine_mean", "top1_real_cosine", "top5_real_cosine_std"]].to_numpy()).all())
    validation["encoder"] = {"name": args.encoder_model, "pooling": "last_hidden_state[:, 0, :]"}
    validation["passed"] = bool(validation["mmd_finite"] and validation["proximity_finite"] and all(x["train_count"] == args.per_class_budget * 4 for x in validation["experiments"].values()))
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    result_rows = train_experiments(datasets, output_dir, args)
    results = pd.DataFrame(result_rows)
    baseline = results.set_index("Experiment")
    deltas = {
        "E2_minus_E1": {"Accuracy": float(baseline.loc["E2", "Accuracy"] - baseline.loc["E1", "Accuracy"]), "Macro_F1": float(baseline.loc["E2", "Macro_F1"] - baseline.loc["E1", "Macro_F1"])},
        "E3_minus_E1": {"Accuracy": float(baseline.loc["E3", "Accuracy"] - baseline.loc["E1", "Accuracy"]), "Macro_F1": float(baseline.loc["E3", "Macro_F1"] - baseline.loc["E1", "Macro_F1"])},
        "E4_minus_E3": {"Accuracy": float(baseline.loc["E4", "Accuracy"] - baseline.loc["E3", "Accuracy"]), "Macro_F1": float(baseline.loc["E4", "Macro_F1"] - baseline.loc["E3", "Macro_F1"])},
        "E4_minus_E2": {"Accuracy": float(baseline.loc["E4", "Accuracy"] - baseline.loc["E2", "Accuracy"]), "Macro_F1": float(baseline.loc["E4", "Macro_F1"] - baseline.loc["E2", "Macro_F1"])},
        "E4_minus_E1": {"Accuracy": float(baseline.loc["E4", "Accuracy"] - baseline.loc["E1", "Accuracy"]), "Macro_F1": float(baseline.loc["E4", "Macro_F1"] - baseline.loc["E1", "Macro_F1"])},
    }
    results.to_csv(output_dir / "pilot_results.csv", index=False)
    (output_dir / "deltas.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    save_excel(output_dir, results, allocation, scores, datasets)
    summary = {"args": vars(args), "tau": tau, "allocation": allocation.to_dict(orient="records"), "pilot_results": result_rows, "deltas": deltas, "e4_beats_e2_and_e3": bool(baseline.loc["E4", "Macro_F1"] > baseline.loc["E2", "Macro_F1"] and baseline.loc["E4", "Macro_F1"] > baseline.loc["E3", "Macro_F1"])}
    (output_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="pilot", choices=["pilot"])
    parser.add_argument("--real-reference", default="qc_train", choices=["qc_train"])
    parser.add_argument("--per-class-budget", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mmd-repeats", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--encoder-model", default=str(ROOT / "cloud" / "model" / "roberta-base"))
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
