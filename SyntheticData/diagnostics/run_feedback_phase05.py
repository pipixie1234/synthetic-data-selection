#!/usr/bin/env python3
"""Create strict ReferenceSeed x ModelSeed downstream metrics for Phase0.5.

No training or generation occurs. Existing frozen checkpoints are evaluated on
each RealReference160, producing reference-conditioned Accuracy/Macro-F1. The
fixed Dev2000 score is also recorded as a model-level audit control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "SyntheticData" / "diagnostics" / "outputs" / "feedback_stability"


def evaluate(model_dir: Path, rows: list[dict], batch_size: int, max_length: int, device: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            block = rows[start:start + batch_size]
            encoded = tokenizer([x["text"] for x in block], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            pred = model(**encoded).logits.argmax(-1).cpu().numpy()
            for row, label in zip(block, pred):
                output.append({"sample_id": row["sample_id"], "ReferenceSeed": row["reference"], "label": int(row["label"]), "prediction": int(label), "ModelSeed": None})
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pd.DataFrame(output)


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    ref_rows = pd.read_csv(out / "reference_samples.csv").to_dict("records")
    refs = sorted({str(x["reference"]) for x in ref_rows})
    model_seeds = [int(x) for x in args.model_seeds]
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions = []
    metric_rows = []
    for seed in model_seeds:
        model_dir = out / "training" / f"synthetic_fixed_model{seed}" / "final"
        result_path = out / "training" / f"synthetic_fixed_model{seed}" / "results.json"
        if not model_dir.exists() or not result_path.exists():
            raise FileNotFoundError(f"Missing Phase0 checkpoint/result for model seed {seed}: {model_dir}")
        model_result = json.loads(result_path.read_text(encoding="utf-8"))
        dev_metrics = model_result.get("metrics", {})
        for ref in refs:
            rows = [x for x in ref_rows if str(x["reference"]) == ref]
            pred = evaluate(model_dir, rows, args.batch_size, args.max_length, device)
            pred["ModelSeed"] = seed
            pred["hard_error"] = pred["prediction"] != pred["label"]
            all_predictions.append(pred)
            metric_rows.append({
                "ReferenceSeed": ref,
                "ModelSeed": seed,
                "Reference_Accuracy": float(accuracy_score(pred["label"], pred["prediction"])),
                "Reference_Macro_F1": float(f1_score(pred["label"], pred["prediction"], average="macro")),
                "Dev2000_Accuracy": float(dev_metrics.get("eval_accuracy", np.nan)),
                "Dev2000_Macro_F1": float(dev_metrics.get("eval_macro_f1", np.nan)),
                "Source": str(result_path),
            })
    pd.DataFrame(metric_rows).to_csv(out / "reference_downstream_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "reference_downstream_predictions.csv", index=False)
    print(f"Wrote {len(metric_rows)} strict ReferenceSeed x ModelSeed metrics")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(DEFAULT_DIR)); p.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--max-length", type=int, default=256); p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
