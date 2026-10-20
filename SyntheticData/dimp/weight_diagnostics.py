"""Diagnostics-only recovery of DIMP sample weights.

This script never changes Phase 1 training outputs.  It loads the frozen
splits and cached QC scores, discovers already-saved DIMP checkpoints, and
recomputes q_i, p_i, and q_i / p_i for the fixed 320-sample training pool.

When a DIMP checkpoint was not saved by the original run, the script writes a
diagnostic CSV with stable sample metadata and explicit ``NOT_AVAILABLE``
status rather than fabricating probabilities or weights.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_dimp_llama_qwen import (
    LABELS,
    LLAMA_SYNTHETIC_PATH,
    QWEN_SYNTHETIC_PATH,
    load_jsonl,
)


DEFAULT_PROTOCOL_DIR = Path(__file__).resolve().parent / "outputs" / "phase1"
DEFAULT_MMD_DIR = Path(__file__).resolve().parents[1] / "MMD" / "outputs" / "distribution_analysis"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "cloud" / "model" / "extra_data" / "phase1_checkpoints"
SOURCES = ("qwen", "llama")
SEEDS = (42, 43, 44)
SCOPES = ("World", "Sports", "Business", "SciTech")
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

SAMPLE_COLUMNS = [
    "sample_id", "source", "label", "text", "quality_score_q",
    "final_p", "final_weight", "diagnostic_status",
]
CLASS_COLUMNS = [
    "Source", "Seed", "Class", "Num_Samples", "Mean_q", "Std_q",
    "Mean_p", "Std_p", "Mean_Weight", "Std_Weight", "Median_Weight",
    "Min_Weight", "Max_Weight", "P25_Weight", "P75_Weight",
    "P90_Weight", "P95_Weight", "ESS", "ESS_Ratio", "Status",
]


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_training_records(protocol_dir: Path, source: str) -> list[dict[str, Any]]:
    """Load fixed train records and assign IDs from immutable source indices."""
    split_ids = _json(protocol_dir / "splits" / f"{source}_train_ids.json")
    data_path = QWEN_SYNTHETIC_PATH if source == "qwen" else LLAMA_SYNTHETIC_PATH
    full_records = load_jsonl(Path(data_path))
    records = []
    for original_index in split_ids:
        record = full_records[int(original_index)]
        records.append({
            "sample_id": f"{source}_{int(original_index):04d}",
            "source": source,
            "original_index": int(original_index),
            "label": int(record["label"]),
            "text": str(record["text"]),
        })
    return records


def load_quality_scores(protocol_dir: Path, source: str) -> list[float]:
    path = protocol_dir / "quality_scores" / f"{source}_train_quality_scores.json"
    values = _json(path)
    return [float(value) for value in values]


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").exists()
        and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
    )


def _checkpoint_metadata(path: Path) -> tuple[float, int]:
    """Return (epoch, step), using trainer_state when available."""
    state_path = path / "trainer_state.json"
    if state_path.exists():
        try:
            state = _json(state_path)
            epoch = state.get("epoch")
            step = int(state.get("global_step", 0) or 0)
            if epoch is not None:
                return float(epoch), step
            history = state.get("log_history", [])
            epochs = [row.get("epoch") for row in history if row.get("epoch") is not None]
            if epochs:
                return float(epochs[-1]), step
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return (float("nan"), int(match.group(1)) if match else 0)


def discover_checkpoints(run_dir: Path) -> dict[str, Any]:
    """Find final and epoch checkpoints without assuming Trainer batch order."""
    candidates: set[Path] = set()
    for candidate in (run_dir, run_dir / "final", run_dir / "final_checkpoint", run_dir / "checkpoint-final"):
        if _is_model_dir(candidate):
            candidates.add(candidate)
    if run_dir.exists():
        for candidate in run_dir.glob("checkpoint-*"):
            if _is_model_dir(candidate):
                candidates.add(candidate)

    metadata = [(path, *_checkpoint_metadata(path)) for path in candidates]
    if not metadata:
        return {"final": None, "epochs": {}, "candidates": []}

    # Highest epoch/step is the final checkpoint.  An explicitly named final
    # directory wins when it exists.
    final_named = [item for item in metadata if item[0].name in {"final", "final_checkpoint", "checkpoint-final"}]
    final = final_named[0][0] if final_named else max(
        metadata,
        key=lambda item: (
            0 if math.isnan(item[1]) else 1,
            item[1] if not math.isnan(item[1]) else -1,
            item[2],
        ),
    )[0]

    epochs: dict[int, Path] = {}
    for path, epoch, step in metadata:
        if math.isnan(epoch):
            continue
        epoch_number = int(round(epoch))
        previous = epochs.get(epoch_number)
        if previous is None or _checkpoint_metadata(path)[1] > _checkpoint_metadata(previous)[1]:
            epochs[epoch_number] = path
    return {"final": final, "epochs": epochs, "candidates": sorted(candidates)}


def _tokenizer_for(checkpoint: Path | None, model_name: str):
    if checkpoint is not None:
        try:
            return AutoTokenizer.from_pretrained(checkpoint)
        except (OSError, ValueError):
            pass
    return AutoTokenizer.from_pretrained(model_name)


def infer_weights(
    checkpoint: Path,
    records: list[dict[str, Any]],
    quality_scores: list[float],
    model_name: str,
    batch_size: int,
    max_length: int,
    gamma: float,
    epsilon: float,
) -> pd.DataFrame:
    """Run deterministic, ordered inference over all records."""
    if len(records) != len(quality_scores):
        raise ValueError(f"records ({len(records)}) and q scores ({len(quality_scores)}) differ")
    tokenizer = _tokenizer_for(checkpoint, model_name)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            texts = [row["text"] for row in batch]
            labels = torch.tensor([row["label"] for row in batch], device=device)
            inputs = tokenizer(
                texts, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            p = probs[torch.arange(len(batch), device=device), labels].cpu().numpy()
            q = np.asarray(quality_scores[start:start + len(batch)], dtype=float)
            weights = (q / np.clip(p, epsilon, None)) ** gamma
            for row, q_value, p_value, weight in zip(batch, q, p, weights):
                rows.append({
                    "sample_id": row["sample_id"], "source": row["source"],
                    "label": row["label"], "text": row["text"],
                    "quality_score_q": float(q_value), "final_p": float(p_value),
                    "final_weight": float(weight), "diagnostic_status": "OK",
                })
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def unavailable_samples(records: list[dict[str, Any]], quality_scores: list[float]) -> pd.DataFrame:
    rows = []
    for record, q_value in zip(records, quality_scores):
        rows.append({
            "sample_id": record["sample_id"], "source": record["source"],
            "label": record["label"], "text": record["text"],
            "quality_score_q": float(q_value), "final_p": np.nan,
            "final_weight": np.nan, "diagnostic_status": "NOT_AVAILABLE",
        })
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def _status_ok(df: pd.DataFrame) -> bool:
    return not df.empty and df["final_weight"].notna().all() and (df["diagnostic_status"] == "OK").all()


def class_weight_stats(df: pd.DataFrame, source: str, seed: int) -> pd.DataFrame:
    rows = []
    status = "OK" if _status_ok(df) else "NOT_AVAILABLE"
    for label, class_name in LABEL_NAMES.items():
        class_df = df[df["label"] == label]
        row: dict[str, Any] = {
            "Source": source.capitalize(), "Seed": seed, "Class": class_name,
            "Num_Samples": len(class_df), "Status": status,
        }
        for key in CLASS_COLUMNS:
            row.setdefault(key, np.nan)
        if len(class_df):
            q = class_df["quality_score_q"].to_numpy(float)
            row.update({
                "Mean_q": float(q.mean()),
                "Std_q": float(q.std(ddof=1)) if len(q) > 1 else 0.0,
            })
        if status == "OK" and len(class_df):
            p = class_df["final_p"].to_numpy(float)
            w = class_df["final_weight"].to_numpy(float)
            ess = float(w.sum() ** 2 / np.square(w).sum()) if np.square(w).sum() else 0.0
            row.update({
                "Mean_q": float(q.mean()), "Std_q": float(q.std(ddof=1)),
                "Mean_p": float(p.mean()), "Std_p": float(p.std(ddof=1)),
                "Mean_Weight": float(w.mean()), "Std_Weight": float(w.std(ddof=1)),
                "Median_Weight": float(np.median(w)), "Min_Weight": float(w.min()),
                "Max_Weight": float(w.max()), "P25_Weight": float(np.percentile(w, 25)),
                "P75_Weight": float(np.percentile(w, 75)), "P90_Weight": float(np.percentile(w, 90)),
                "P95_Weight": float(np.percentile(w, 95)), "ESS": ess,
                "ESS_Ratio": float(ess / len(w)),
            })
        rows.append(row)
    return pd.DataFrame(rows, columns=CLASS_COLUMNS)


def aggregate_class_stats(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_map = {
        "Mean_DIMP_Weight": "Mean_Weight", "Median_DIMP_Weight": "Median_Weight",
        "P95_DIMP_Weight": "P95_Weight", "ESS_Ratio": "ESS_Ratio",
    }
    for (source, class_name), group in stats.groupby(["Source", "Class"], sort=False):
        ok = group[group["Status"] == "OK"]
        row: dict[str, Any] = {"Source": source, "Class": class_name, "Num_Seeds": len(ok), "Status": "OK" if len(ok) else "NOT_AVAILABLE"}
        for source_metric, input_metric in metric_map.items():
            row[source_metric] = float(ok[input_metric].mean()) if len(ok) else np.nan
            row[f"{source_metric}_Std_Across_Seeds"] = float(ok[input_metric].std(ddof=1)) if len(ok) > 1 else (0.0 if len(ok) == 1 else np.nan)
        for output, input_metric in {"Mean_q": "Mean_q", "Mean_p": "Mean_p"}.items():
            source_values = group[input_metric].dropna()
            row[output] = float(source_values.mean()) if len(source_values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_sample_weights(sample_frames: dict[tuple[str, int], pd.DataFrame], source: str) -> pd.DataFrame:
    frames = [df for (frame_source, _), df in sample_frames.items() if frame_source == source and _status_ok(df)]
    if len(frames) < 1:
        return pd.DataFrame()
    merged = frames[0][["sample_id", "source", "label", "text", "quality_score_q"]].copy()
    merged["q_mean"] = 0.0
    merged["final_p_mean"] = 0.0
    merged["final_weight_mean"] = 0.0
    merged["final_weight_std"] = 0.0
    merged["_p_values"] = [[] for _ in range(len(merged))]
    merged["_w_values"] = [[] for _ in range(len(merged))]
    epoch_numbers = sorted({
        int(match.group(1))
        for frame in frames
        for column in frame.columns
        for match in [re.match(r"epoch_(\d+)_weight$", column)]
        if match
    })
    for epoch in epoch_numbers:
        merged[f"epoch{epoch}_weight_mean"] = 0.0
    for frame in frames:
        indexed = frame.set_index("sample_id")
        merged["q_mean"] += merged["sample_id"].map(indexed["quality_score_q"]).to_numpy(float)
        merged["final_p_mean"] += merged["sample_id"].map(indexed["final_p"]).to_numpy(float)
        merged["_p_values"] = [values + [float(indexed.loc[sample_id, "final_p"])] for values, sample_id in zip(merged["_p_values"], merged["sample_id"])]
        merged["_w_values"] = [values + [float(indexed.loc[sample_id, "final_weight"])] for values, sample_id in zip(merged["_w_values"], merged["sample_id"])]
        for epoch in epoch_numbers:
            column = f"epoch_{epoch}_weight"
            if column in indexed:
                merged[f"epoch{epoch}_weight_mean"] += merged["sample_id"].map(indexed[column]).to_numpy(float)
    n = len(frames)
    merged["q_mean"] /= n
    merged["final_p_mean"] /= n
    merged["final_weight_mean"] = merged["_w_values"].map(np.mean)
    merged["final_weight_std"] = merged["_w_values"].map(lambda values: np.std(values, ddof=1) if len(values) > 1 else 0.0)
    for epoch in epoch_numbers:
        column = f"epoch_{epoch}_weight"
        available = sum(column in frame.columns for frame in frames)
        merged[f"epoch{epoch}_weight_mean"] /= available
    return merged.drop(columns=["_p_values", "_w_values"])


def _top_bottom(aggregate: pd.DataFrame, output_dir: Path, source: str) -> None:
    columns = ["rank", "sample_id", "source", "label", "text", "q_mean", "final_p_mean", "final_weight_mean", "final_weight_std"]
    columns += sorted(column for column in aggregate.columns if re.fullmatch(r"epoch\d+_weight_mean", column))
    if aggregate.empty:
        empty = pd.DataFrame(columns=columns + ["status"])
        empty.to_csv(output_dir / f"{source}_top20_weights.csv", index=False)
        empty.to_csv(output_dir / f"{source}_bottom20_weights.csv", index=False)
        return
    base = aggregate.sort_values("final_weight_mean", ascending=False).copy()
    top = base.head(20).copy()
    bottom = base.tail(20).sort_values("final_weight_mean", ascending=True).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    bottom.insert(0, "rank", range(1, len(bottom) + 1))
    top[columns].to_csv(output_dir / f"{source}_top20_weights.csv", index=False)
    bottom[columns].to_csv(output_dir / f"{source}_bottom20_weights.csv", index=False)


def load_mmd_relationship(mmd_dir: Path) -> pd.DataFrame:
    path = mmd_dir / "mmd_dimp_relationship.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_mmd_weight_relationship(aggregated: pd.DataFrame, mmd_dir: Path) -> pd.DataFrame:
    mmd = load_mmd_relationship(mmd_dir)
    rows = []
    for _, row in aggregated.iterrows():
        source, class_name = row["Source"], row["Class"]
        match = mmd[(mmd.get("Source", pd.Series(dtype=str)) == source) & (mmd.get("Scope", pd.Series(dtype=str)).replace({"Sci/Tech": "SciTech"}) == class_name)] if not mmd.empty else pd.DataFrame()
        mmd_row = match.iloc[0] if not match.empty else pd.Series(dtype=float)
        rows.append({
            "Source": source, "Class": class_name,
            "MMD_to_Real": mmd_row.get("MMD_to_Real", np.nan),
            "Mean_q": row.get("Mean_q", np.nan), "Mean_p": row.get("Mean_p", np.nan),
            "Mean_DIMP_Weight": row.get("Mean_DIMP_Weight", np.nan),
            "Median_DIMP_Weight": row.get("Median_DIMP_Weight", np.nan),
            "P95_DIMP_Weight": row.get("P95_DIMP_Weight", np.nan),
            "ESS_Ratio": row.get("ESS_Ratio", np.nan),
            "CE_F1": mmd_row.get("CE_F1", np.nan), "DIMP_F1": mmd_row.get("DIMP_F1", np.nan),
            "DIMP_Gain_F1": mmd_row.get("DIMP_Gain_F1", np.nan), "Status": row.get("Status", "NOT_AVAILABLE"),
        })
    return pd.DataFrame(rows)


def correlation_table(relationship: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["Mean_DIMP_Weight", "Median_DIMP_Weight", "P95_DIMP_Weight", "ESS_Ratio", "DIMP_Gain_F1"]:
        valid = relationship[["MMD_to_Real", metric]].dropna()
        result: dict[str, Any] = {"Metric_X": "MMD_to_Real", "Metric_Y": metric, "N": len(valid), "Analysis": "exploratory"}
        if len(valid) >= 3 and valid["MMD_to_Real"].nunique() > 1 and valid[metric].nunique() > 1:
            pearson = pearsonr(valid["MMD_to_Real"], valid[metric])
            spearman = spearmanr(valid["MMD_to_Real"], valid[metric])
            result.update({"Pearson_r": float(pearson.statistic), "Pearson_p": float(pearson.pvalue), "Spearman_rho": float(spearman.statistic), "Spearman_p": float(spearman.pvalue)})
        else:
            result.update({"Pearson_r": np.nan, "Pearson_p": np.nan, "Spearman_rho": np.nan, "Spearman_p": np.nan})
        rows.append(result)
    return pd.DataFrame(rows)


def gradient_norm_report(protocol_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        for seed in SEEDS:
            run_dir = protocol_dir / (f"B3_Qwen320_DIMP_seed{seed}" if source == "qwen" else f"C3_Llama320_DIMP_seed{seed}")
            values = []
            for path in [run_dir / "epoch_diagnostics.json", run_dir / "trainer_state.json"]:
                if not path.exists():
                    continue
                try:
                    payload = _json(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                items = payload if isinstance(payload, list) else payload.get("log_history", []) if isinstance(payload, dict) else []
                for item in items:
                    if isinstance(item, dict) and item.get("grad_norm") is not None:
                        values.append({"Epoch": item.get("epoch", np.nan), "Global_Step": item.get("step", item.get("global_step", np.nan)), "Grad_Norm": item["grad_norm"]})
            if values:
                grad_values = np.asarray([float(value["Grad_Norm"]) for value in values], dtype=float)
                ordered = sorted(values, key=lambda value: (
                    -float(value["Epoch"]) if pd.notna(value["Epoch"]) else float("inf"),
                    -float(value["Global_Step"]) if pd.notna(value["Global_Step"]) else float("inf"),
                ))
                rows.append({
                    "Source": source.capitalize(), "Seed": seed,
                    "mean_grad_norm": float(grad_values.mean()),
                    "max_grad_norm": float(grad_values.max()),
                    "final_grad_norm": float(ordered[0]["Grad_Norm"]),
                    "num_observations": len(values), "Status": "OK",
                })
            else:
                rows.append({
                    "Source": source.capitalize(), "Seed": seed,
                    "mean_grad_norm": np.nan, "max_grad_norm": np.nan,
                    "final_grad_norm": np.nan, "num_observations": 0,
                    "Status": "NOT_AVAILABLE",
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "gradient_norm_summary.csv", index=False)
    return df


def validation_report(sample_frames: dict[tuple[str, int], pd.DataFrame], output_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    all_frames = []
    for (source, seed), df in sample_frames.items():
        report[f"{source}_seed{seed}_count"] = int(len(df))
        report[f"{source}_seed{seed}_count_is_320"] = bool(len(df) == 320)
        report[f"{source}_seed{seed}_duplicate_sample_id"] = bool(df["sample_id"].duplicated().any())
        report[f"{source}_seed{seed}_status"] = "OK" if _status_ok(df) else "NOT_AVAILABLE"
        for key in ("quality_score_q", "final_p", "final_weight"):
            values = pd.to_numeric(df[key], errors="coerce")
            report[f"{source}_seed{seed}_{key}_nan_count"] = int(values.isna().sum())
            report[f"{source}_seed{seed}_{key}_inf_count"] = int(np.isinf(values.fillna(0)).sum())
        if _status_ok(df):
            weights = df["final_weight"].to_numpy(float)
            for threshold in (5, 10, 20, 50):
                report[f"{source}_seed{seed}_weight_gt_{threshold}"] = int((weights > threshold).sum())
        all_frames.append(df)
    for source in SOURCES:
        ids = [set(sample_frames[(source, seed)]["sample_id"]) for seed in SEEDS]
        report[f"sample_id_consistency_{source}"] = bool(ids and all(ids[0] == item for item in ids[1:]))
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    for key in ("quality_score_q", "final_p", "final_weight"):
        values = pd.to_numeric(combined.get(key, pd.Series(dtype=float)), errors="coerce")
        report[f"{key}_nan_count"] = int(values.isna().sum())
        report[f"{key}_inf_count"] = int(np.isinf(values.fillna(0)).sum())
    report["nan_count"] = sum(value for key, value in report.items() if key.endswith("_nan_count"))
    report["inf_count"] = sum(value for key, value in report.items() if key.endswith("_inf_count"))
    (output_dir / "weight_diagnostics_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_report(stats: pd.DataFrame, correlation: pd.DataFrame, gradients: pd.DataFrame, validation: dict[str, Any], output_dir: Path) -> None:
    print("\nWeight diagnostics report")
    for source in SOURCES:
        for seed in SEEDS:
            print(f"{source}/seed{seed}: {validation.get(f'{source}_seed{seed}_count', 0)} samples; {validation.get(f'{source}_seed{seed}_status', 'NOT_AVAILABLE')}")
        source_stats = stats[(stats["Source"].str.lower() == source) & (stats["Status"] == "OK")]
        if not source_stats.empty:
            print(f"{source.capitalize()} class weights:")
            print(source_stats[["Class", "Mean_Weight", "Median_Weight", "P95_Weight"]].to_string(index=False))
    ok_stats = stats[stats["Status"] == "OK"]
    if not ok_stats.empty:
        max_row = ok_stats.loc[ok_stats["Mean_Weight"].idxmax()]
        min_row = ok_stats.loc[ok_stats["ESS_Ratio"].idxmin()]
        print(f"Largest mean weight: {max_row['Source']} / {max_row['Class']}")
        print(f"Lowest ESS ratio: {min_row['Source']} / {min_row['Class']}")
    top_bottom_files = list(output_dir.glob("*_top20_weights.csv")) + list(output_dir.glob("*_bottom20_weights.csv"))
    top_bottom_ready = bool(top_bottom_files) and all(pd.read_csv(path).shape[0] == 20 for path in top_bottom_files)
    print(f"Top/Bottom exports: {'available' if top_bottom_ready else 'NOT_AVAILABLE'}")
    print("Correlation (exploratory):")
    print(correlation.to_string(index=False))
    print(f"Gradient norm: {'available' if (gradients['Status'] == 'OK').any() else 'NOT_AVAILABLE'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover DIMP sample-level weight diagnostics from existing checkpoints")
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--mmd-dir", type=Path, default=DEFAULT_MMD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROTOCOL_DIR / "weight_diagnostics")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_frames: dict[tuple[str, int], pd.DataFrame] = {}
    stats_frames = []
    for source in SOURCES:
        records = load_training_records(args.protocol_dir, source)
        quality_scores = load_quality_scores(args.protocol_dir, source)
        if len(records) != 320:
            raise ValueError(f"Expected 320 {source} training records, got {len(records)}")
        for seed in SEEDS:
            run_name = f"B3_Qwen320_DIMP_seed{seed}" if source == "qwen" else f"C3_Llama320_DIMP_seed{seed}"
            checkpoint_locations = [args.checkpoint_dir / run_name, args.protocol_dir / run_name]
            found = {"final": None, "epochs": {}, "candidates": []}
            for checkpoint_location in checkpoint_locations:
                found = discover_checkpoints(checkpoint_location)
                if found["final"] is not None:
                    break
            if found["final"] is None:
                frame = unavailable_samples(records, quality_scores)
                print(f"{run_name}: no DIMP checkpoint found; writing NOT_AVAILABLE")
            else:
                frame = infer_weights(found["final"], records, quality_scores, args.model_name, args.batch_size, args.max_length, args.gamma, args.epsilon)
                for epoch, checkpoint in sorted(found["epochs"].items()):
                    epoch_frame = infer_weights(checkpoint, records, quality_scores, args.model_name, args.batch_size, args.max_length, args.gamma, args.epsilon)
                    frame[f"epoch_{epoch}_p"] = epoch_frame["final_p"]
                    frame[f"epoch_{epoch}_weight"] = epoch_frame["final_weight"]
            sample_frames[(source, seed)] = frame
            frame.to_csv(args.output_dir / f"{source}_seed{seed}_samples.csv", index=False)
            stats_frames.append(class_weight_stats(frame, source, seed))

    by_seed = pd.concat(stats_frames, ignore_index=True)
    by_seed.to_csv(args.output_dir / "class_weight_stats_by_seed.csv", index=False)
    aggregated = aggregate_class_stats(by_seed)
    aggregated.to_csv(args.output_dir / "class_weight_stats_aggregated.csv", index=False)

    for source in SOURCES:
        aggregate = aggregate_sample_weights(sample_frames, source)
        _top_bottom(aggregate, args.output_dir, source)

    relationship = build_mmd_weight_relationship(aggregated, args.mmd_dir)
    relationship.to_csv(args.output_dir / "mmd_dimp_weight_relationship.csv", index=False)
    correlation = correlation_table(relationship)
    correlation.to_csv(args.output_dir / "weight_correlation_results.csv", index=False)
    gradients = gradient_norm_report(args.protocol_dir, args.output_dir)
    validation = validation_report(sample_frames, args.output_dir)
    print_report(by_seed, correlation, gradients, validation, args.output_dir)


if __name__ == "__main__":
    main()
