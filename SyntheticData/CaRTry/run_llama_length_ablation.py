"""Llama-only length-controlled ablation with fixed Qwen TopK data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "SyntheticData" / "exp"
DEFAULT_OUT = ROOT / "SyntheticData" / "CaRTry" / "outputs" / "llama_length_ablation"
sys.path.insert(0, str(EXP))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402


METHODS = ["Random40", "Shortest40", "Length-Matched-Random40", "TopK40"]


def n_tokens(row: dict[str, Any]) -> int:
    return len(str(row["text"]).split())


def records_with_score(rows: list[dict[str, Any]], score_map: dict[str, float], method: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["proximity_score"] = float(score_map.get(row["sample_id"], np.nan))
        item["selection_method"] = method
        item.pop("original_index", None)
        out.append(item)
    return out


def length_matched_random(pool: list[dict[str, Any]], target: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Match TopK's four candidate-length quartile counts without scores."""
    frame = pd.DataFrame([{"sample_id": x["sample_id"], "length": n_tokens(x)} for x in pool])
    target_frame = pd.DataFrame([{"sample_id": x["sample_id"], "length": n_tokens(x)} for x in target])
    frame["quartile"] = pd.qcut(frame["length"].rank(method="first"), 4, labels=[0, 1, 2, 3]).astype(int)
    target_ids = set(target_frame.sample_id)
    target_frame = target_frame.merge(frame[["sample_id", "quartile"]], on="sample_id", how="left")
    desired = target_frame.quartile.value_counts().reindex(range(4), fill_value=0).astype(int).to_dict()
    rng = random.Random(seed)
    selected = []
    for q in range(4):
        candidates = [x for x in pool if int(frame.loc[frame.sample_id == x["sample_id"], "quartile"].iloc[0]) == q]
        count = int(desired[q])
        if count > len(candidates):
            raise RuntimeError(f"Length-matched quota {count} exceeds quartile {q} pool {len(candidates)}")
        selected.extend(rng.sample(candidates, count))
    if len(selected) != 40 or len({x["sample_id"] for x in selected}) != 40:
        raise RuntimeError("Length-matched selection is not exactly 40 unique samples")
    return selected


def make_sets(data: dict[str, list[dict[str, Any]]], scores: pd.DataFrame, topk: list[dict[str, Any]], out: Path) -> dict[str, list[dict[str, Any]]]:
    score_map = scores.set_index("sample_id")["top5_real_cosine_mean"].to_dict()
    qwen_fixed = [x for x in topk if x["source"] == "qwen"]
    topk_llama = [x for x in topk if x["source"] == "llama"]
    sets = {}
    for method in METHODS:
        llama_selected = []
        for label in range(4):
            pool = [x for x in data["llama_train"] if int(x["label"]) == label]
            target = [x for x in topk_llama if int(x["label"]) == label]
            if method == "Random40":
                selected = random.Random(42 + label * 1009).sample(pool, 40)
            elif method == "Shortest40":
                selected = sorted(pool, key=lambda x: (n_tokens(x), x["sample_id"]))[:40]
            elif method == "Length-Matched-Random40":
                selected = length_matched_random(pool, target, 4200 + label)
            else:
                selected = target
            llama_selected.extend(selected)
        rows = records_with_score(qwen_fixed + llama_selected, score_map, method)
        rows.sort(key=lambda x: (int(x["label"]), x["sample_id"]))
        if len(rows) != 320 or len({x["sample_id"] for x in rows}) != 320:
            raise RuntimeError(f"{method}: invalid total size")
        if any(sum(x["source"] == source and int(x["label"]) == label for x in rows) != 40 for source in ("qwen", "llama") for label in range(4)):
            raise RuntimeError(f"{method}: invalid source/class allocation")
        sets[method] = rows
        pd.DataFrame(rows).to_csv(out / f"llama_ablation_{method}.csv", index=False)
    return sets


def length_summary(sets: dict[str, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]], topk: list[dict[str, Any]], out: Path) -> pd.DataFrame:
    rows = []
    for method, records in sets.items():
        for label in range(4):
            cand = [n_tokens(x) for x in data["llama_train"] if int(x["label"]) == label]
            selected = [n_tokens(x) for x in records if x["source"] == "llama" and int(x["label"]) == label]
            top = [n_tokens(x) for x in topk if x["source"] == "llama" and int(x["label"]) == label]
            rows.append({"Method": method, "Class": curation.LABEL_NAMES[label], "Candidate_Median": np.median(cand), "Selected_Median": np.median(selected), "TopK_Median": np.median(top), "Candidate_Mean": np.mean(cand), "Selected_Mean": np.mean(selected), "TopK_Mean": np.mean(top), "Selected_P10": np.quantile(selected, .1), "Selected_P90": np.quantile(selected, .9)})
    frame = pd.DataFrame(rows); frame.to_csv(out / "llama_length_ablation_summary.csv", index=False); return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(DEFAULT_OUT)); p.add_argument("--validation-dir", default=str(EXP / "outputs" / "curation_validation")); p.add_argument("--model-seed", type=int, default=42); p.add_argument("--evaluate-legacy400", action="store_true"); p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--bert-model", default="bert-base-uncased"); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data = curation.load_frozen_data(); validation_dir = Path(args.validation_dir); scores = pd.read_csv(validation_dir / "sample_proximity_scores.csv"); topk = pd.read_csv(validation_dir / "e2_uniform_topk_fixed.csv").to_dict("records")
    sets = make_sets(data, scores, topk, out); summary = length_summary(sets, data, topk, out)
    (out / "protocol.json").write_text(json.dumps({"methods": METHODS, "scope": "Llama40 replacement only; Qwen fixed to E2 TopK40", "model_seed": args.model_seed, "final_holdout_evaluated": False}, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared {len(sets)} Llama-only ablation datasets under {out}"); print(summary.to_string(index=False)); return
    dev = pd.read_csv(validation_dir / "dev2000.csv").to_dict("records"); legacy = data["real_test"] if args.evaluate_legacy400 else None
    train_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False)
    result_rows = []
    for method, records in sets.items():
        run_name = f"LlamaAblation_{method}_model{args.model_seed}"
        result = validation.train_one(records, dev, legacy, None, out, run_name, args.model_seed, train_args)
        result_rows.append(validation.metric_row(method, "llama_only", args.model_seed, result, out / "training" / run_name))
    results = pd.DataFrame(result_rows); results.to_csv(out / "llama_ablation_run_results.csv", index=False)
    merged = results[["Experiment", "Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1", "Legacy400_Accuracy", "Legacy400_Macro_F1"]]
    merged.to_markdown(out / "llama_ablation_results.md", index=False, floatfmt=".6f")
    with pd.ExcelWriter(out / "llama_ablation_results.xlsx", engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Results", index=False); summary.to_excel(writer, sheet_name="Length_Summary", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
