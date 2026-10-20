"""Multi-seed validation of Uniform+Random (E1) versus Uniform+TopK (E2).

The curation inputs are frozen Phase 1 pools.  Dev2000 is created from the
original AG News train split after excluding the real samples already used by
the project, and is used only as an evaluation set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "SyntheticData"
PROTOCOL_DIR = SYNTHETIC_DIR / "SmaExp" / "outputs" / "protocol"
SPLITS_DIR = SYNTHETIC_DIR / "dimp" / "outputs" / "phase1" / "splits"
OUTPUT_DIR = SYNTHETIC_DIR / "exp" / "outputs" / "curation_validation"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

sys.path.insert(0, str(SYNTHETIC_DIR / "exp"))
import run_distribution_curation as curation  # noqa: E402


def load_arrow_split(split: str) -> list[dict[str, Any]]:
    """Read the locally cached official AG News Arrow split."""
    from datasets import Dataset

    arrow_name = f"ag_news-{split}.arrow"
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets"))
    matches = list(datasets_cache.rglob(arrow_name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find cached {arrow_name}. Download/cache fancyzhx/ag_news first."
        )
    dataset = Dataset.from_file(str(matches[0]))
    return [
        {"original_dataset_index": i, "text": row["text"], "label": int(row["label"])}
        for i, row in enumerate(dataset)
    ]


def load_protocol_real() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        curation.load_jsonl(PROTOCOL_DIR / "real_train.jsonl"),
        curation.load_jsonl(PROTOCOL_DIR / "real_test.jsonl"),
    )


def audit_with_official_test(real_train_rows: list[dict[str, Any]], legacy_rows: list[dict[str, Any]], official_test: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    legacy_indices = [int(row["source_index"]) for row in legacy_rows]
    official_by_index = {row["original_dataset_index"]: row for row in official_test}
    matches = 0
    for row in legacy_rows:
        official = official_by_index.get(int(row["source_index"]))
        if official is not None and int(row["label"]) == official["label"] and str(row["text"]) == str(official["text"]):
            matches += 1
    if matches != len(legacy_rows) or len(set(legacy_indices)) != len(legacy_indices):
        raise RuntimeError("Legacy RealTest400 could not be proven to be an official test subset")
    legacy_set = set(legacy_indices)
    final_holdout = [
        {**row, "sample_id": f"official_test_{row['original_dataset_index']}", "source": "official_test_holdout"}
        for row in official_test
        if row["original_dataset_index"] not in legacy_set
    ]
    audit = {
        "legacy_test400_source": "fancyzhx/ag_news official test split, source_index 0-based",
        "legacy_test400_count": len(legacy_rows),
        "legacy_test400_indices": sorted(legacy_indices),
        "final_holdout_source": "fancyzhx/ag_news official test split minus legacy RealTest400",
        "final_holdout_count": len(final_holdout),
        "final_holdout_indices": [row["original_dataset_index"] for row in final_holdout],
        "overlap_count": 0,
        "legacy_rows_verified_against_official_test": matches,
    }
    return audit, final_holdout


def build_dev2000(real_train_rows: list[dict[str, Any]], official_train: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size % 4 != 0:
        raise ValueError("--dev-size must be divisible by four")
    excluded = {
        int(row["source_index"])
        for row in real_train_rows
        if row.get("source_split") == "train"
    }
    available = [row for row in official_train if row["original_dataset_index"] not in excluded]
    per_class = size // 4
    rng = random.Random(seed)
    selected = []
    for label in range(4):
        pool = [row for row in available if int(row["label"]) == label]
        if len(pool) < per_class:
            raise RuntimeError(f"Not enough development samples for class {label}")
        selected.extend(rng.sample(pool, per_class))
    selected.sort(key=lambda row: row["original_dataset_index"])
    return [
        {
            "sample_id": f"dev_{row['original_dataset_index']}",
            "original_dataset_index": row["original_dataset_index"],
            "text": row["text"],
            "label": int(row["label"]),
        }
        for row in selected
    ]


def validate_dev_exclusion(dev: list[dict[str, Any]], real_train_rows: list[dict[str, Any]], legacy_rows: list[dict[str, Any]]) -> None:
    dev_ids = {row["original_dataset_index"] for row in dev}
    used_train = {int(row["source_index"]) for row in real_train_rows if row.get("source_split") == "train"}
    if dev_ids & used_train:
        raise RuntimeError("Dev2000 overlaps existing real train/QC pool")
    if len(dev) != 2000 or Counter(row["label"] for row in dev) != Counter({0: 500, 1: 500, 2: 500, 3: 500}):
        raise RuntimeError("Dev2000 is not exactly class-balanced")
    if {row["text"] for row in dev} & {row["text"] for row in legacy_rows}:
        raise RuntimeError("Dev2000 text overlaps legacy RealTest400")


def save_protocol(output_dir: Path, data: dict[str, list[dict[str, Any]]], audit: dict[str, Any], dev: list[dict[str, Any]], e1_sets: dict[int, list[dict[str, Any]]], e2: list[dict[str, Any]], model_seeds: list[int], selection_seeds: list[int]) -> None:
    (output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(dev).to_csv(output_dir / "dev2000.csv", index=False)
    (output_dir / "dev2000_ids.json").write_text(json.dumps([row["original_dataset_index"] for row in dev], indent=2), encoding="utf-8")
    pd.DataFrame(e2).to_csv(output_dir / "e2_uniform_topk_fixed.csv", index=False)
    for selection_seed, rows in e1_sets.items():
        pd.DataFrame(rows).to_csv(output_dir / f"e1_random_sel{selection_seed}.csv", index=False)
    manifest = {
        "real_reference": {"count": len(data["real_reference"]), "sample_ids": [x["sample_id"] for x in data["real_reference"]]},
        "dev2000": {"count": len(dev), "sample_ids": [x["sample_id"] for x in dev], "original_dataset_indices": [x["original_dataset_index"] for x in dev]},
        "legacy400": {"count": len(data["real_test"]), "sample_ids": [x["sample_id"] for x in data["real_test"]], "provenance": audit["legacy_test400_source"]},
        "final_holdout": {"count": audit["final_holdout_count"], "source": audit["final_holdout_source"], "original_dataset_indices": audit["final_holdout_indices"]},
        "synthetic_candidate_pools": {"qwen_ids": [x["sample_id"] for x in data["qwen_train"]], "llama_ids": [x["sample_id"] for x in data["llama_train"]]},
        "e1_selection_seeds": selection_seeds,
        "e1_selection_files": [f"e1_random_sel{s}.csv" for s in selection_seeds],
        "e2_selection_seed": "deterministic_topk",
        "e2_sample_ids": [x["sample_id"] for x in e2],
        "model_seeds": model_seeds,
        "dev_selection_seed": 2026,
        "top_k": 5,
        "selection_inputs": {"encoder": "roberta-base", "pooling": "last_hidden_state[:, 0, :]", "real_reference": "qc_train"},
    }
    (output_dir / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def metric_row(experiment: str, selection_seed: str, model_seed: int, result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    primary = result["metrics"]
    per_class = result["per_class_metrics"]
    row = {
        "Experiment": experiment,
        "Selection_Seed": selection_seed,
        "Model_Seed": model_seed,
        "Train_Count": result["train_count"],
        "Dev_Accuracy": primary["eval_accuracy"],
        "Dev_Macro_F1": primary["eval_macro_f1"],
        "Dev_World_F1": per_class.get("World", {}).get("f1-score", np.nan),
        "Dev_Sports_F1": per_class.get("Sports", {}).get("f1-score", np.nan),
        "Dev_Business_F1": per_class.get("Business", {}).get("f1-score", np.nan),
        "Dev_SciTech_F1": per_class.get("Sci/Tech", {}).get("f1-score", np.nan),
        "Run_Dir": str(run_dir),
    }
    legacy = result.get("additional_evaluations", {}).get("legacy400")
    if legacy:
        row.update({"Legacy400_Accuracy": legacy["metrics"]["accuracy"], "Legacy400_Macro_F1": legacy["metrics"]["macro_f1"]})
        for name, key in [("World", "Legacy400_World_F1"), ("Sports", "Legacy400_Sports_F1"), ("Business", "Legacy400_Business_F1"), ("Sci/Tech", "Legacy400_SciTech_F1")]:
            row[key] = legacy["per_class_metrics"].get(name, {}).get("f1-score", np.nan)
    else:
        for key in ["Legacy400_Accuracy", "Legacy400_Macro_F1", "Legacy400_World_F1", "Legacy400_Sports_F1", "Legacy400_Business_F1", "Legacy400_SciTech_F1"]:
            row[key] = np.nan
    final = result.get("additional_evaluations", {}).get("final_holdout")
    if final:
        row.update({"FinalHoldout_Accuracy": final["metrics"]["accuracy"], "FinalHoldout_Macro_F1": final["metrics"]["macro_f1"]})
    return row


def train_one(train_records: list[dict[str, Any]], dev: list[dict[str, Any]], legacy: list[dict[str, Any]] | None, final_holdout: list[dict[str, Any]] | None, output_dir: Path, run_name: str, model_seed: int, args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(SYNTHETIC_DIR / "dimp"))
    from phase1_trainers import Phase1Trainer

    result_path = output_dir / "training" / run_name / "results.json"
    if result_path.exists():
        try:
            cached = json.loads(result_path.read_text())
            if cached.get("train_count") == len(train_records) and cached.get("seed") == model_seed and "additional_evaluations" in cached:
                print(f"Reusing {run_name}")
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    trainer = Phase1Trainer(model_name=args.bert_model, num_labels=4, loss_type="ce")
    extra = {}
    if args.evaluate_legacy400 and legacy is not None:
        extra["legacy400"] = legacy
    if args.evaluate_final_holdout and final_holdout is not None:
        extra["final_holdout"] = final_holdout
    return trainer.train(
        train_records,
        dev,
        quality_scores=None,
        output_dir=output_dir / "training",
        run_name=run_name,
        seed=model_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_checkpoints=False,
        additional_eval_records=extra,
    )


def mean_std(frame: pd.DataFrame, columns: list[str], prefix: str = "") -> pd.DataFrame:
    rows = []
    for col in columns:
        rows.append({"Metric": col, "Mean": float(frame[col].mean()), "Std": float(frame[col].std(ddof=1)) if len(frame) > 1 else 0.0, "N": int(frame[col].count())})
    return pd.DataFrame(rows)


def jaccard(a: set[str], b: set[str]) -> float:
    return float(len(a & b) / len(a | b)) if a | b else 1.0


def ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return (float(values.mean()), float(values.mean()))
    margin = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return float(values.mean() - margin), float(values.mean() + margin)


def analyze(results: pd.DataFrame, e1_sets: dict[int, list[dict[str, Any]]], e2: list[dict[str, Any]], scores: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    e1 = results[results.Experiment == "E1"].copy()
    e2_results = results[results.Experiment == "E2"].copy()
    metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]
    main_rows = []
    for label, frame in [("E1_Random_25Runs", e1), ("E2_TopK_5Runs", e2_results)]:
        for metric in metrics:
            main_rows.append({"Experiment": label, "Metric": metric, "Mean": frame[metric].mean(), "Std": frame[metric].std(ddof=1), "N": len(frame)})
    main = pd.DataFrame(main_rows)

    selection_rows = []
    for seed, frame in e1.groupby("Selection_Seed"):
        for metric in metrics:
            selection_rows.append({"Selection_Seed": seed, "Metric": metric, "Mean": frame[metric].mean(), "Std": frame[metric].std(ddof=1), "N": len(frame)})
    selection = pd.DataFrame(selection_rows)

    model_rows = []
    for seed, frame in results.groupby(["Experiment", "Model_Seed"]):
        exp, model_seed = seed
        subset = frame if exp == "E1" else frame
        for metric in metrics:
            model_rows.append({"Experiment": exp, "Model_Seed": model_seed, "Metric": metric, "Mean": subset[metric].mean(), "Std": subset[metric].std(ddof=1) if len(subset) > 1 else 0.0, "N": len(subset)})
    model = pd.DataFrame(model_rows)

    paired_rows = []
    for model_seed in sorted(e2_results.Model_Seed.unique()):
        e2_row = e2_results[e2_results.Model_Seed == model_seed].iloc[0]
        random_rows = e1[e1.Model_Seed == model_seed]
        for metric in metrics:
            delta = float(e2_row[metric] - random_rows[metric].mean())
            paired_rows.append({"Model_Seed": model_seed, "Metric": metric, "E2": e2_row[metric], "RandomMean": random_rows[metric].mean(), "Delta": delta})
    paired = pd.DataFrame(paired_rows)
    summary_rows = []
    for metric in metrics:
        values = paired[paired.Metric == metric].Delta.to_numpy(dtype=float)
        lo, hi = ci95(values)
        summary_rows.append({"Metric": metric, "Delta_Mean": values.mean(), "Delta_Std": values.std(ddof=1), "CI95_Low": lo, "CI95_High": hi, "Positive_Count": int((values > 0).sum()), "N": len(values)})
    paired_summary = pd.DataFrame(summary_rows)
    paired = paired.merge(paired_summary, on="Metric", how="left")

    # Balanced two-way variance decomposition over E1's 5 x 5 grid.
    variance_rows = []
    for metric in ["Dev_Accuracy", "Dev_Macro_F1"]:
        grid = e1.pivot(index="Selection_Seed", columns="Model_Seed", values=metric)
        grand = float(grid.to_numpy().mean())
        selection_means = grid.mean(axis=1)
        model_means = grid.mean(axis=0)
        selection_var = float(((selection_means - grand) ** 2).mean())
        model_var = float(((model_means - grand) ** 2).mean())
        residual = grid.copy()
        for s in grid.index:
            for m in grid.columns:
                residual.loc[s, m] = grid.loc[s, m] - selection_means.loc[s] - model_means.loc[m] + grand
        residual_var = float((residual.to_numpy() ** 2).mean())
        total = selection_var + model_var + residual_var
        variance_rows.append({"Metric": metric, "Selection_Variance": selection_var, "Model_Variance": model_var, "Residual_Variance": residual_var, "Selection_Fraction": selection_var / total if total else 0.0, "Model_Fraction": model_var / total if total else 0.0, "Residual_Fraction": residual_var / total if total else 0.0})
    variance = pd.DataFrame(variance_rows)

    class_rows = []
    for label, col in [("World", "Dev_World_F1"), ("Sports", "Dev_Sports_F1"), ("Business", "Dev_Business_F1"), ("SciTech", "Dev_SciTech_F1")]:
        random_means = e1.groupby("Model_Seed")[col].mean()
        topk = e2_results[col]
        deltas = topk.to_numpy() - random_means.loc[e2_results.Model_Seed].to_numpy()
        all_comparisons = []
        for _, topk_row in e2_results.iterrows():
            all_comparisons.extend((topk_row[col] > e1[e1.Model_Seed == topk_row.Model_Seed][col]).astype(int).tolist())
        class_rows.append({"Class": label, "Random_Mean_F1": e1[col].mean(), "Random_Std": e1[col].std(ddof=1), "TopK_Mean_F1": topk.mean(), "TopK_Std": topk.std(ddof=1), "Delta": deltas.mean(), "Delta_Std": deltas.std(ddof=1), "Win_Rate": np.mean(all_comparisons)})
    per_class = pd.DataFrame(class_rows)

    random_ids = {seed: {x["sample_id"] for x in rows} for seed, rows in e1_sets.items()}
    e2_ids = {x["sample_id"] for x in e2}
    overlap_rows = []
    for a in sorted(random_ids):
        row = {"Selection_Seed": a}
        for b in sorted(random_ids):
            row[f"Random_{b}"] = jaccard(random_ids[a], random_ids[b])
        row["E2_TopK"] = jaccard(random_ids[a], e2_ids)
        overlap_rows.append(row)
    overlap = pd.DataFrame(overlap_rows)

    score_map = scores.set_index("sample_id")["top5_real_cosine_mean"]
    prox_rows = [{"Dataset": "E2_fixed_topk", "Selection_Seed": "deterministic_topk", "Proximity_Mean": float(np.mean([score_map[x["sample_id"]] for x in e2])), "Proximity_Std": float(np.std([score_map[x["sample_id"]] for x in e2], ddof=1)), "Proximity_Median": float(np.median([score_map[x["sample_id"]] for x in e2])), "Dev_Macro_F1": e2_results.Dev_Macro_F1.mean()}]
    for seed, rows in e1_sets.items():
        values = np.asarray([score_map[x["sample_id"]] for x in rows], dtype=float)
        prox_rows.append({"Dataset": "E1_random", "Selection_Seed": seed, "Proximity_Mean": values.mean(), "Proximity_Std": values.std(ddof=1), "Proximity_Median": np.median(values), "Dev_Macro_F1": e1[e1.Selection_Seed == str(seed)].Dev_Macro_F1.mean()})
    proximity = pd.DataFrame(prox_rows)
    if len(proximity) >= 3:
        proximity["Proximity_vs_MacroF1_Pearson"] = proximity["Proximity_Mean"].corr(proximity["Dev_Macro_F1"])
    else:
        proximity["Proximity_vs_MacroF1_Pearson"] = np.nan

    overall_win_rows = []
    for metric in ["Dev_Accuracy", "Dev_Macro_F1"]:
        wins = []
        for _, topk_row in e2_results.iterrows():
            wins.extend((topk_row[metric] > e1[e1.Model_Seed == topk_row.Model_Seed][metric]).astype(int).tolist())
        overall_win_rows.append({"Metric": metric, "E2_vs_Random_Wins": sum(wins), "Comparisons": len(wins), "E2_vs_Random_Win_Rate": float(np.mean(wins))})
    win_rate = pd.DataFrame(overall_win_rows)

    sheets = {"Main_Results": main, "Selection_Seed_Results": selection, "Model_Seed_Results": model, "Paired_Deltas": paired, "Paired_Delta_Summary": paired_summary, "Per_Class_F1": per_class, "Variance_Decomposition": variance, "Dataset_Overlap": overlap, "Proximity_Analysis": proximity, "Win_Rate": win_rate}
    with pd.ExcelWriter(output_dir / "e1_e2_validation_results.xlsx", engine="openpyxl") as writer:
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet[:31], index=False)

    report_lines = [
        "E1/E2 multi-seed validation report (objective summary)",
        f"E2 mean Dev Accuracy={e2_results.Dev_Accuracy.mean():.6f}, Macro-F1={e2_results.Dev_Macro_F1.mean():.6f}",
        f"E1 random25 mean Dev Accuracy={e1.Dev_Accuracy.mean():.6f}, Macro-F1={e1.Dev_Macro_F1.mean():.6f}",
        f"E2 - RandomMean Accuracy={(e2_results.Dev_Accuracy.mean() - e1.Dev_Accuracy.mean()):.6f}, Macro-F1={(e2_results.Dev_Macro_F1.mean() - e1.Dev_Macro_F1.mean()):.6f}",
        f"Positive paired model-seed deltas: Accuracy={int((paired_summary[paired_summary.Metric == 'Dev_Accuracy'].Positive_Count.iloc[0]))}/5, Macro-F1={int((paired_summary[paired_summary.Metric == 'Dev_Macro_F1'].Positive_Count.iloc[0]))}/5",
        f"E2 vs random win rate: Accuracy={win_rate.loc[win_rate.Metric == 'Dev_Accuracy', 'E2_vs_Random_Win_Rate'].iloc[0]:.6f}, Macro-F1={win_rate.loc[win_rate.Metric == 'Dev_Macro_F1', 'E2_vs_Random_Win_Rate'].iloc[0]:.6f}",
        f"Random selection variance (std of selection means): Accuracy={e1.groupby('Selection_Seed').Dev_Accuracy.mean().std(ddof=1):.6f}, Macro-F1={e1.groupby('Selection_Seed').Dev_Macro_F1.mean().std(ddof=1):.6f}",
        f"Model training variance (average within-selection std): Accuracy={e1.groupby('Selection_Seed').Dev_Accuracy.std(ddof=1).mean():.6f}, Macro-F1={e1.groupby('Selection_Seed').Dev_Macro_F1.std(ddof=1).mean():.6f}",
        f"SciTech paired delta mean={per_class.loc[per_class.Class == 'SciTech', 'Delta'].iloc[0]:.6f}, std={per_class.loc[per_class.Class == 'SciTech', 'Delta_Std'].iloc[0]:.6f}, win_rate={per_class.loc[per_class.Class == 'SciTech', 'Win_Rate'].iloc[0]:.6f}",
        f"Sports delta mean={per_class.loc[per_class.Class == 'Sports', 'Delta'].iloc[0]:.6f}; Business delta mean={per_class.loc[per_class.Class == 'Business', 'Delta'].iloc[0]:.6f}",
        "Interpretation: descriptive/exploratory only; no automatic claim of success.",
    ]
    (output_dir / "e1_e2_validation_analysis.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return sheets


def load_prepared_selection(output_dir: Path) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]], pd.DataFrame]:
    """Load selection artifacts created by --prepare-only."""
    scores = pd.read_csv(output_dir / "sample_proximity_scores.csv")
    e1_sets = {}
    for path in sorted(output_dir.glob("e1_random_sel*.csv")):
        seed = int(path.stem.replace("e1_random_sel", ""))
        e1_sets[seed] = pd.read_csv(path).to_dict("records")
    e2 = pd.read_csv(output_dir / "e2_uniform_topk_fixed.csv").to_dict("records")
    if len(e1_sets) != 5 or len(e2) != 320:
        raise RuntimeError("Prepared selection artifacts are incomplete")
    return e1_sets, e2, scores


def merge_shards(output_dir: Path) -> Path:
    shard_files = sorted((output_dir / "shards").glob("shard_*/run_results.csv"))
    if len(shard_files) == 0:
        raise FileNotFoundError(f"No shard result files found under {output_dir / 'shards'}")
    results = pd.concat([pd.read_csv(path) for path in shard_files], ignore_index=True)
    expected = 30
    keys = results[["Experiment", "Selection_Seed", "Model_Seed"]].astype(str).drop_duplicates()
    if len(results) != expected or len(keys) != expected:
        raise RuntimeError(f"Expected {expected} unique runs, found {len(results)} rows/{len(keys)} keys")
    results.to_csv(output_dir / "run_results.csv", index=False)
    e1_sets, e2, scores = load_prepared_selection(output_dir)
    analyze(results, e1_sets, e2, scores, output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--dev-seed", type=int, default=2026)
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--real-reference", choices=["qc_train"], default="qc_train")
    parser.add_argument("--evaluate-legacy400", action="store_true")
    parser.add_argument("--evaluate-final-holdout", action="store_true", default=False)
    parser.add_argument("--no-final-holdout", dest="evaluate_final_holdout", action="store_false")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--encoder-model", default=str(ROOT / "cloud" / "model" / "roberta-base"))
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--prepare-only", action="store_true", help="Create split/selection/cache artifacts without training")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based training shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of training shards")
    parser.add_argument("--merge-shards", action="store_true", help="Merge completed shard CSVs and produce final analysis")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if len(args.selection_seeds) != 5 or len(args.model_seeds) != 5:
        raise ValueError("This protocol requires exactly five selection seeds and five model seeds")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        return merge_shards(output_dir)
    if args.num_shards < 1 or (args.shard_index is not None and not 0 <= args.shard_index < args.num_shards):
        raise ValueError("Invalid --shard-index/--num-shards")
    real_train_rows, legacy_rows = load_protocol_real()
    official_train = load_arrow_split("train")
    official_test = load_arrow_split("test")
    audit, final_holdout = audit_with_official_test(real_train_rows, legacy_rows, official_test)
    dev = build_dev2000(real_train_rows, official_train, args.dev_size, args.dev_seed)
    validate_dev_exclusion(dev, real_train_rows, legacy_rows)

    data = curation.load_frozen_data()
    embeddings = curation.load_or_compute_embeddings(data, output_dir, args.encoder_model, args.embedding_batch_size)
    scores = curation.proximity_scores(data, embeddings, args.top_k)
    uniform_allocation = pd.DataFrame({"label": [0, 1, 2, 3], "Qwen_Count": [40] * 4, "Llama_Count": [40] * 4})
    e2 = curation.select_dataset(data, scores, uniform_allocation, "uniform_topk", 80, 42)
    e1_sets = {seed: curation.select_dataset(data, scores, uniform_allocation, "uniform_random", 80, seed) for seed in args.selection_seeds}
    prepared = (
        (output_dir / "protocol_manifest.json").exists()
        and (output_dir / "dev2000.csv").exists()
        and (output_dir / "e2_uniform_topk_fixed.csv").exists()
        and all((output_dir / f"e1_random_sel{s}.csv").exists() for s in args.selection_seeds)
    )
    if not prepared or args.prepare_only:
        save_protocol(output_dir, data, audit, dev, e1_sets, e2, args.model_seeds, args.selection_seeds)
    scores = curation.mark_selected(scores, e2, e2)
    if args.prepare_only or not (output_dir / "sample_proximity_scores.csv").exists():
        scores.to_csv(output_dir / "sample_proximity_scores.csv", index=False)

    if args.prepare_only:
        print(f"Prepared validation artifacts under {output_dir}")
        return output_dir

    legacy = data["real_test"] if args.evaluate_legacy400 else None
    jobs = []
    for selection_seed, selected in e1_sets.items():
        for model_seed in args.model_seeds:
            jobs.append(("E1", str(selection_seed), model_seed, selected, f"E1_sel{selection_seed}_model{model_seed}"))
    for model_seed in args.model_seeds:
        jobs.append(("E2", "deterministic_topk", model_seed, e2, f"E2_deterministic_topk_model{model_seed}"))
    if args.shard_index is not None:
        jobs = jobs[args.shard_index::args.num_shards]
        train_output_dir = output_dir / "shards" / f"shard_{args.shard_index}"
    else:
        train_output_dir = output_dir
    all_rows = []
    for experiment, selection_seed, model_seed, selected, run_name in jobs:
        result = train_one(selected, dev, legacy, final_holdout, train_output_dir, run_name, model_seed, args)
        all_rows.append(metric_row(experiment, selection_seed, model_seed, result, train_output_dir / "training" / run_name))
    results = pd.DataFrame(all_rows)
    if args.shard_index is not None:
        train_output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(train_output_dir / "run_results.csv", index=False)
        print(f"Completed shard {args.shard_index}/{args.num_shards}: {len(results)} runs")
    else:
        results.to_csv(output_dir / "run_results.csv", index=False)
        analyze(results, e1_sets, e2, scores, output_dir)
    return output_dir


if __name__ == "__main__":
    run(parse_args())
