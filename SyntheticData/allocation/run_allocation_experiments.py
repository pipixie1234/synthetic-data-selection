"""Frozen-TopK allocation experiments for AG News.

Assumed allocation grid: for each class, one Qwen-heavy (60/20) and one
Llama-heavy (20/60) variant, with all other classes fixed at 40/40. The
Joint-Top80 condition selects the top 80 from the combined source pool per
class. All selection uses the frozen E2 proximity score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "SyntheticData" / "exp"
VAL = EXP / "outputs" / "curation_validation"
OUT = ROOT / "SyntheticData" / "allocation" / "outputs" / "allocation_validation"
sys.path.insert(0, str(EXP))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}
MODEL_SEEDS = [42, 43, 44]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT)); p.add_argument("--validation-dir", default=str(VAL))
    p.add_argument("--model-seeds", nargs="+", type=int, default=MODEL_SEEDS)
    p.add_argument("--prepare-only", action="store_true"); p.add_argument("--evaluate-legacy400", action="store_true")
    p.add_argument("--bert-model", default="bert-base-uncased"); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    data = curation.load_frozen_data(); scores = pd.read_csv(Path(args.validation_dir) / "sample_proximity_scores.csv")
    if not np.isfinite(scores["top5_real_cosine_mean"].to_numpy(float)).all(): raise RuntimeError("Frozen proximity cache contains non-finite scores")
    return data, scores


def top_sorted(records: list[dict[str, Any]], score_map: dict[str, float]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda x: (-float(score_map[x["sample_id"]]), x["sample_id"]))


def output_records(rows: list[dict[str, Any]], score_map: dict[str, float], condition: str, allocation: dict[int, tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row); item["proximity_score"] = float(score_map[row["sample_id"]]); item["selection_method"] = condition; item["allocation_condition"] = condition; item.pop("original_index", None); out.append(item)
    out.sort(key=lambda x: (int(x["label"]), x["sample_id"]))
    return out


def make_conditions(data: dict[str, list[dict[str, Any]]], scores: pd.DataFrame, out: Path) -> dict[str, list[dict[str, Any]]]:
    score_map = scores.set_index("sample_id")["top5_real_cosine_mean"].to_dict(); conditions: dict[str, list[dict[str, Any]]] = {}
    for focal in range(4):
        for qwen_count, llama_count, direction in [(60, 20, "Qwen60_Llama20"), (20, 60, "Qwen20_Llama60")]:
            name = f"{LABEL_NAMES[focal]}_{direction}"; selected = []
            allocation = {}
            for label in range(4):
                allocation[label] = (qwen_count, llama_count) if label == focal else (40, 40)
                qpool = top_sorted([x for x in data["qwen_train"] if int(x["label"]) == label], score_map)[: allocation[label][0]]
                lpool = top_sorted([x for x in data["llama_train"] if int(x["label"]) == label], score_map)[: allocation[label][1]]
                selected.extend(qpool + lpool)
            rows = output_records(selected, score_map, name)
            validate_dataset(rows, allocation, name); conditions[name] = rows; pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    selected = []
    for label in range(4):
        pool = [x for source in ("qwen", "llama") for x in data[f"{source}_train"] if int(x["label"]) == label]
        selected.extend(top_sorted(pool, score_map)[:80])
    rows = output_records(selected, score_map, "Joint-Top80"); validate_joint(rows); conditions["Joint-Top80"] = rows; pd.DataFrame(rows).to_csv(out / "Joint-Top80.csv", index=False)
    return conditions


def validate_dataset(rows: list[dict[str, Any]], allocation: dict[int, tuple[int, int]], name: str) -> None:
    if len(rows) != 320 or len({x["sample_id"] for x in rows}) != 320: raise RuntimeError(f"{name}: invalid total or duplicate IDs")
    for label, (q, l) in allocation.items():
        if sum(x["source"] == "qwen" and int(x["label"]) == label for x in rows) != q: raise RuntimeError(f"{name}: Qwen allocation mismatch")
        if sum(x["source"] == "llama" and int(x["label"]) == label for x in rows) != l: raise RuntimeError(f"{name}: Llama allocation mismatch")


def validate_joint(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 320 or len({x["sample_id"] for x in rows}) != 320: raise RuntimeError("Joint-Top80: invalid total or duplicate IDs")
    if any(sum(int(x["label"]) == label for x in rows) != 80 for label in range(4)): raise RuntimeError("Joint-Top80: class allocation mismatch")


def make_manifest(conditions: dict[str, list[dict[str, Any]]], out: Path) -> pd.DataFrame:
    rows = []
    for name, records in conditions.items():
        for label, cls in LABEL_NAMES.items():
            rows.append({"Condition": name, "Class": cls, "Total": sum(int(x["label"]) == label for x in records), "Qwen": sum(x["source"] == "qwen" and int(x["label"]) == label for x in records), "Llama": sum(x["source"] == "llama" and int(x["label"]) == label for x in records), "Unique_IDs": len({x["sample_id"] for x in records if int(x["label"]) == label})})
    frame = pd.DataFrame(rows); frame.to_csv(out / "allocation_dataset_manifest.csv", index=False); return frame


def train(args: argparse.Namespace, conditions: dict[str, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]], out: Path) -> pd.DataFrame:
    dev = pd.read_csv(Path(args.validation_dir) / "dev2000.csv").to_dict("records"); legacy = data["real_test"] if args.evaluate_legacy400 else None; trainer_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False); rows = []
    for condition, records in conditions.items():
        for seed in args.model_seeds:
            run_name = f"{condition}_model{seed}"; result = validation.train_one(records, dev, legacy, None, out, run_name, seed, trainer_args); rows.append(validation.metric_row(condition, "frozen_topk_allocation", seed, result, out / "training" / run_name))
    frame = pd.DataFrame(rows); frame.to_csv(out / "allocation_run_results.csv", index=False); return frame


def write_summary(results: pd.DataFrame, out: Path, validation_dir: Path) -> None:
    metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]
    summary = results.groupby("Experiment").agg(N_Runs=("Model_Seed", "count"), **{f"{m}_Mean": (m, "mean") for m in metrics}, **{f"{m}_Std": (m, "std") for m in metrics}).reset_index().rename(columns={"Experiment": "Condition"})
    summary.to_csv(out / "allocation_summary.csv", index=False)
    top = pd.read_csv(validation_dir / "run_results.csv"); top = top[top.Experiment == "E2"]
    rows = []
    for condition in results.Experiment.unique():
        if condition == "Joint-Top80": label = condition
        else: label = condition
        for metric in metrics:
            deltas = []
            for seed in sorted(results.Model_Seed.unique()):
                a = results[(results.Experiment == condition) & (results.Model_Seed == seed)][metric].iloc[0]
                t = top[top.Model_Seed == seed][metric].iloc[0]
                deltas.append(float(a - t))
            rows.append({"Condition": label, "Metric": metric, "Mean_Delta_vs_TopK": np.mean(deltas), "Std_Delta_vs_TopK": np.std(deltas, ddof=1), "Positive_Seeds": int(sum(x > 0 for x in deltas)), "Seed42": deltas[0], "Seed43": deltas[1], "Seed44": deltas[2]})
    deltas = pd.DataFrame(rows); deltas.to_csv(out / "allocation_paired_deltas.csv", index=False)
    with pd.ExcelWriter(out / "allocation_results.xlsx", engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Runs", index=False); summary.to_excel(writer, sheet_name="Summary", index=False); deltas.to_excel(writer, sheet_name="Deltas_vs_TopK", index=False)


def main() -> None:
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.model_seeds != MODEL_SEEDS: raise ValueError("This protocol requires model seeds 42,43,44")
    data, scores = load_inputs(args); conditions = make_conditions(data, scores, out); manifest = make_manifest(conditions, out)
    (out / "allocation_protocol.json").write_text(json.dumps({"frozen_selector": "E2 TopK proximity", "class_shift_ratios": ["60/20", "20/60"], "focal_classes": list(LABEL_NAMES.values()), "joint_condition": "top80 from combined Qwen+Llama per class", "model_seeds": args.model_seeds, "new_runs": 27, "final_holdout_evaluated": False}, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(manifest.to_string(index=False)); print(f"Prepared {len(conditions)} allocation datasets under {out}"); return
    results = train(args, conditions, data, out); write_summary(results, out, Path(args.validation_dir)); print(results.to_string(index=False))


if __name__ == "__main__":
    main()
