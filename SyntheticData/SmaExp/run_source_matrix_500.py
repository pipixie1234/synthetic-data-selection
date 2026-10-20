"""Run a 4x4 train-source/test-source BERT matrix on the expanded pools.

The expanded Qwen/Llama candidate pools have 500 records per class but no
independent synthetic test split. This script creates one deterministic,
non-overlapping 400/100 split per source/class, keeps the existing real
protocol untouched, and evaluates all train-source x test-source pairs:
real, qwen, llama, mix.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "SyntheticData"
PROTOCOL = SYN / "SmaExp" / "outputs" / "protocol"
DEFAULT_POOL = ROOT / "cloud" / "model" / "extra_data" / "candidate_pool" / "outputs" / "expanded_candidate_pool.csv"
DEFAULT_OUT = SYN / "SmaExp" / "outputs" / "comparison_500_matrix"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}
SOURCES = ("real", "qwen", "llama", "mix")

sys.path.insert(0, str(SYN / "dimp"))
from phase1_trainers import Phase1Trainer  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(row.get("sample_id", f"{source}_{i}")),
            "text": str(row["text"]),
            "label": int(row["label"]),
            "source": source,
        }
        for i, row in enumerate(rows)
    ]


def load_splits(pool_path: Path, split_seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    real_train = normalize(load_jsonl(PROTOCOL / "real_train.jsonl"), "real")
    real_test = normalize(load_jsonl(PROTOCOL / "real_test.jsonl"), "real")
    frame = pd.read_csv(pool_path)
    required = {"sample_id", "source", "label", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Expanded pool is missing columns: {sorted(missing)}")
    if set(frame.source.unique()) != {"qwen", "llama"}:
        raise ValueError(f"Expected qwen/llama sources, got {sorted(frame.source.unique())}")
    if frame.duplicated("sample_id").any() or frame.text.duplicated().any():
        raise ValueError("Expanded pool contains duplicate sample IDs or exact duplicate text")

    train: dict[str, list[dict[str, Any]]] = {"real": real_train}
    test: dict[str, list[dict[str, Any]]] = {"real": real_test}
    split_manifest: dict[str, Any] = {
        "split_seed": split_seed,
        "pool_path": str(pool_path.resolve()),
        "synthetic_policy": "500/source/class -> 400 train + 100 test, deterministic per source/class",
        "real_policy": "existing SmaExp protocol: 200 train + 400 test",
        "mix_policy": "Qwen200 + Llama200 train/class; Qwen100 + Llama100 test/class",
        "counts": {},
    }
    for source, offset in (("qwen", 0), ("llama", 100000)):
        src_train: list[dict[str, Any]] = []
        src_test: list[dict[str, Any]] = []
        for label in range(4):
            group = frame[(frame.source == source) & (frame.label == label)].sort_values("sample_id")
            if len(group) != 500:
                raise ValueError(f"Expected 500 {source}/{LABEL_NAMES[label]} candidates, got {len(group)}")
            rows = group[["sample_id", "text", "label"]].to_dict("records")
            random.Random(split_seed + offset + label).shuffle(rows)
            src_train.extend(normalize(rows[:400], source))
            src_test.extend(normalize(rows[400:], source))
        train[source] = src_train
        test[source] = src_test
        split_manifest["counts"][source] = {"train": len(src_train), "test": len(src_test)}

    train["mix"] = []
    test["mix"] = []
    for label in range(4):
        train["mix"].extend([r for r in train["qwen"] if r["label"] == label][:200])
        train["mix"].extend([r for r in train["llama"] if r["label"] == label][:200])
        test["mix"].extend([r for r in test["qwen"] if r["label"] == label])
        test["mix"].extend([r for r in test["llama"] if r["label"] == label])
    split_manifest["counts"]["real"] = {"train": len(train["real"]), "test": len(test["real"])}
    split_manifest["counts"]["mix"] = {"train": len(train["mix"]), "test": len(test["mix"])}
    for source in SOURCES:
        if len({r["sample_id"] for r in train[source]}) != len(train[source]):
            raise ValueError(f"Duplicate train IDs in {source}")
        if len({r["sample_id"] for r in test[source]}) != len(test[source]):
            raise ValueError(f"Duplicate test IDs in {source}")
    for source in ("qwen", "llama"):
        if {r["sample_id"] for r in train[source]} & {r["sample_id"] for r in test[source]}:
            raise ValueError(f"Synthetic train/test leakage in {source}")
    return {"train": train, "test": test}, split_manifest


def train_cell(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], args: argparse.Namespace, name: str, seed: int) -> dict[str, Any]:
    result_path = Path(args.output_dir) / "training" / name / "results.json"
    if result_path.exists():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("train_count") == len(train_rows) and cached.get("test_count") == len(test_rows) and cached.get("seed") == seed:
                print(f"Reusing {name} seed={seed}")
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    trainer = Phase1Trainer(model_name=args.bert_model, num_labels=4, loss_type="ce")
    result = trainer.train(
        train_rows,
        test_rows,
        quality_scores=None,
        output_dir=Path(args.output_dir) / "training",
        run_name=name,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_checkpoints=False,
    )
    return result


def flatten_result(result: dict[str, Any], train_source: str, test_source: str, seed: int) -> dict[str, Any]:
    m = result["metrics"]
    return {
        "Train_Source": train_source,
        "Test_Source": test_source,
        "Model_Seed": seed,
        "Train_Count": result["train_count"],
        "Test_Count": result["test_count"],
        "Accuracy": m.get("eval_accuracy", np.nan),
        "Macro_F1": m.get("eval_macro_f1", np.nan),
        "Loss": m.get("eval_loss", np.nan),
        "Run": result.get("run", f"train_{train_source}_test_{test_source}_seed{seed}"),
    }


def write_workbook(rows: list[dict[str, Any]], manifest: dict[str, Any], out: Path) -> None:
    raw = pd.DataFrame(rows).sort_values(["Train_Source", "Test_Source", "Model_Seed"])
    matrix = raw.groupby(["Train_Source", "Test_Source"], as_index=False).Accuracy.mean().pivot(index="Train_Source", columns="Test_Source", values="Accuracy").reindex(index=SOURCES, columns=SOURCES)
    f1_matrix = raw.groupby(["Train_Source", "Test_Source"], as_index=False).Macro_F1.mean().pivot(index="Train_Source", columns="Test_Source", values="Macro_F1").reindex(index=SOURCES, columns=SOURCES)
    loss_matrix = raw.groupby(["Train_Source", "Test_Source"], as_index=False).Loss.mean().pivot(index="Train_Source", columns="Test_Source", values="Loss").reindex(index=SOURCES, columns=SOURCES)
    grouped = raw.groupby(["Train_Source", "Test_Source"], as_index=False).agg(
        Accuracy_Mean=("Accuracy", "mean"), Accuracy_Std=("Accuracy", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        Macro_F1_Mean=("Macro_F1", "mean"), Macro_F1_Std=("Macro_F1", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        Loss_Mean=("Loss", "mean"), Loss_Std=("Loss", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0), N=("Model_Seed", "count"),
    )
    design_rows = [
        {"Item": "Purpose", "Value": "Train-source x test-source cross-domain BERT matrix on expanded Qwen/Llama candidate pool"},
        {"Item": "Training sources", "Value": "real=existing 200; qwen=400/class; llama=400/class; mix=Qwen200+Llama200/class"},
        {"Item": "Testing sources", "Value": "real=existing 400; qwen=100/class holdout; llama=100/class holdout; mix=Qwen100+Llama100/class"},
        {"Item": "Synthetic split", "Value": manifest["synthetic_policy"]},
        {"Item": "Leakage check", "Value": "Synthetic train/test sample IDs are disjoint; real protocol reused unchanged"},
        {"Item": "BERT", "Value": str(manifest["bert_model"])},
        {"Item": "Seeds", "Value": ", ".join(map(str, manifest["model_seeds"]))},
        {"Item": "Interpretation", "Value": "This is a source-domain compatibility diagnostic, not a selector/allocation experiment"},
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="Cell_Results", index=False)
        grouped.to_excel(writer, sheet_name="Mean_Std", index=False)
        matrix.to_excel(writer, sheet_name="Accuracy")
        f1_matrix.to_excel(writer, sheet_name="Macro_F1")
        loss_matrix.to_excel(writer, sheet_name="Loss")
        pd.DataFrame(design_rows).to_excel(writer, sheet_name="Design", index=False)
        for ws in writer.book.worksheets:
            for column_cells in ws.columns:
                width = min(max(len(str(c.value or "")) for c in column_cells) + 2, 48)
                ws.column_dimensions[column_cells[0].column_letter].width = width
    raw.to_csv(out.with_suffix(".csv"), index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool-csv", default=str(DEFAULT_POOL))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--summary", default=str(SYN / "SmaExp" / "outputs" / "visualizations" / "comparison_summary_500_matrix.xlsx"))
    p.add_argument("--bert-model", default=os.environ.get("BERT_MODEL", "bert-base-uncased"))
    p.add_argument("--model-seeds", nargs="+", type=int, default=[42])
    p.add_argument("--split-seed", type=int, default=2028)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this matrix on the current HAMI node")
    splits, manifest = load_splits(Path(args.pool_csv), args.split_seed)
    manifest.update({"bert_model": args.bert_model, "model_seeds": args.model_seeds, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_length": args.max_length})
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for seed in args.model_seeds:
        for train_source in SOURCES:
            for test_source in SOURCES:
                name = f"train_{train_source}_test_{test_source}_seed{seed}"
                result = train_cell(splits["train"][train_source], splits["test"][test_source], args, name, seed)
                rows.append(flatten_result(result, train_source, test_source, seed))
                pd.DataFrame(rows).to_csv(out_dir / "cell_results_partial.csv", index=False)
    write_workbook(rows, manifest, Path(args.summary))
    print(f"Wrote {len(rows)} matrix cells to {args.summary}")


if __name__ == "__main__":
    main()
