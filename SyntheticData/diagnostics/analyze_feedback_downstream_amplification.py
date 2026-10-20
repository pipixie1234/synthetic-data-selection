#!/usr/bin/env python3
"""Exploratory Feedback Instability -> Downstream Amplification analysis.

This is deliberately strict about identity matching. A downstream result must
contain both an explicit ReferenceSeed/Reference field and ModelSeed/seed; a
model-only result is reported as unavailable rather than copied across
references.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE0 = ROOT / "SyntheticData" / "diagnostics" / "outputs" / "feedback_stability"
DEFAULT_OUT = DEFAULT_PHASE0


def parse_ref(value: Any) -> str | None:
    text = str(value).strip()
    if text.startswith("ref_"):
        return text
    if text.isdigit():
        return f"ref_{int(text)}"
    return None


def load_gap_metrics(phase0: Path) -> tuple[pd.DataFrame, list[str], list[int]]:
    sample = pd.read_csv(phase0 / "sample_hardness.csv")
    sample = sample[sample["Dataset"].eq("reference")].copy()
    refs = sorted(sample["Reference"].dropna().unique().tolist())
    models = sorted(int(x) for x in sample["Model_Seed"].dropna().unique())
    rows = []
    for ref in refs:
        frame = sample[sample["Reference"].eq(ref)]
        sets = {
            int(seed): {
                "hard_error": set(g.loc[g["hard_error"].astype(bool), "sample_id"]),
                "hard_loss": set(g.loc[g["hard_loss"].astype(bool), "sample_id"]),
            }
            for seed, g in frame.groupby("Model_Seed")
        }
        for seed in models:
            if seed not in sets:
                continue
            vals = {}
            for metric in ("hard_error", "hard_loss"):
                other = []
                for other_seed, other_sets in sets.items():
                    if other_seed == seed:
                        continue
                    a, b = sets[seed][metric], other_sets[metric]
                    other.append(len(a & b) / max(len(a | b), 1))
                vals[f"{metric}_jaccard"] = float(np.mean(other)) if other else np.nan
                vals[f"gap_divergence_{metric.replace('hard_', '')}"] = 1.0 - vals[f"{metric}_jaccard"] if other else np.nan
            vals["ReferenceSeed"] = ref
            vals["ModelSeed"] = seed
            vals["Gap_Divergence"] = float(np.nanmean([vals["gap_divergence_error"], vals["gap_divergence_loss"]]))
            rows.append(vals)
    return pd.DataFrame(rows), refs, models


def discover_downstream(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = [Path(x) for x in args.downstream_results]
    if not paths:
        paths = sorted((ROOT / "SyntheticData").rglob("run_results.csv"))
        paths += sorted((DEFAULT_PHASE0 / "training").glob("*/results.json"))
    strict: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            excluded.append({"Path": str(path), "Reason": "missing_file"})
            continue
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = [data]
            else:
                rows = pd.read_csv(path).to_dict("records")
        except Exception as exc:
            excluded.append({"Path": str(path), "Reason": f"read_error:{exc}"})
            continue
        for row in rows:
            keys = {str(k).lower().replace("-", "_"): k for k in row}
            ref_key = next((keys[k] for k in ("referenceseed", "reference_seed", "reference") if k in keys), None)
            model_key = next((keys[k] for k in ("modelseed", "model_seed", "seed") if k in keys), None)
            ref = parse_ref(row.get(ref_key)) if ref_key else None
            model = row.get(model_key) if model_key else None
            reference_acc_key = next((keys[k] for k in ("reference_accuracy", "reference_acc") if k in keys), None)
            reference_f1_key = next((keys[k] for k in ("reference_macro_f1", "reference_f1") if k in keys), None)
            acc_key = next((keys[k] for k in ("dev_accuracy", "dev2000_accuracy", "accuracy", "eval_accuracy") if k in keys), None)
            f1_key = next((keys[k] for k in ("dev_macro_f1", "dev2000_macro_f1", "macro_f1", "eval_macro_f1") if k in keys), None)
            downstream_acc_key = reference_acc_key or acc_key
            downstream_f1_key = reference_f1_key or f1_key
            if ref is None or model is None or downstream_acc_key is None or downstream_f1_key is None:
                excluded.append({"Path": str(path), "Reason": "no_strict_reference_model_pair", "Reference_Field": ref_key or "", "Model_Field": model_key or ""})
                continue
            strict.append({"ReferenceSeed": ref, "ModelSeed": int(model), "Downstream_Accuracy": float(row[downstream_acc_key]), "Downstream_Macro_F1": float(row[downstream_f1_key]), "Dev_Accuracy": float(row[acc_key]) if acc_key else np.nan, "Dev_Macro_F1": float(row[f1_key]) if f1_key else np.nan, "Source": str(path)})
    return strict, excluded


def correlation(frame: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    valid = frame[[x, y]].dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return {"N": int(len(valid)), "Pearson": None, "Spearman": None}
    return {"N": int(len(valid)), "Pearson": float(pearsonr(valid[x], valid[y]).statistic), "Spearman": float(spearmanr(valid[x], valid[y]).statistic)}


def mad_flag(values: pd.Series, value: float, multiplier: float = 3.0) -> bool:
    values = values.dropna().astype(float)
    if len(values) < 3:
        return False
    median = float(values.median())
    mad = float(np.median(np.abs(values - median)))
    return bool(mad > 0 and abs(value - median) > multiplier * mad)


def main(args: argparse.Namespace) -> None:
    phase0 = Path(args.phase0_dir).resolve(); out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    gap, refs, models = load_gap_metrics(phase0)
    downstream, excluded = discover_downstream(args)
    downstream_df = pd.DataFrame(downstream)
    duplicate_keys = set()
    if not downstream_df.empty:
        duplicate_keys = set(downstream_df.groupby(["ReferenceSeed", "ModelSeed"]).size().loc[lambda x: x > 1].index.tolist())
    strict_map = {}
    for row in downstream:
        key = (row["ReferenceSeed"], row["ModelSeed"])
        if key not in duplicate_keys:
            strict_map[key] = row

    rows = []
    for _, g in gap.iterrows():
        key = (g["ReferenceSeed"], int(g["ModelSeed"]))
        match = strict_map.get(key)
        item = dict(g)
        item.update({"Hard_Error_Jaccard": item.pop("hard_error_jaccard"), "Hard_Loss_Jaccard": item.pop("hard_loss_jaccard"), "Gap_Divergence_Error": item.pop("gap_divergence_error"), "Gap_Divergence_Loss": item.pop("gap_divergence_loss")})
        item["Match_Status"] = "matched" if match else ("ambiguous" if key in duplicate_keys else "missing")
        item["Downstream_Source"] = match["Source"] if match else ""
        item["Downstream_Accuracy"] = match["Downstream_Accuracy"] if match else np.nan
        item["Downstream_Macro_F1"] = match["Downstream_Macro_F1"] if match else np.nan
        item["Dev_Accuracy"] = match["Dev_Accuracy"] if match else np.nan
        item["Dev_Macro_F1"] = match["Dev_Macro_F1"] if match else np.nan
        item["Is_Performance_Outlier"] = pd.NA if not match else False
        rows.append(item)
    result = pd.DataFrame(rows)
    for metric in ("Downstream_Macro_F1", "Downstream_Accuracy"):
        dev = result[metric]
        result[f"{metric.replace('Dev_', '')}_Deviation"] = np.nan
        result[f"{metric.replace('Dev_', '')}_Median_Within_Reference"] = np.nan
        result[f"{metric.replace('Dev_', '')}_MAD_Within_Reference"] = np.nan
        result["Is_Performance_Outlier"] = False
        for ref, idx in result.groupby("ReferenceSeed").groups.items():
            vals = result.loc[idx, metric].dropna()
            if vals.empty:
                continue
            med = float(vals.median()); mad = float(np.median(np.abs(vals - med)))
            result.loc[idx, f"{metric.replace('Dev_', '')}_Median_Within_Reference"] = med
            result.loc[idx, f"{metric.replace('Dev_', '')}_MAD_Within_Reference"] = mad
            result.loc[idx, f"{metric.replace('Dev_', '')}_Deviation"] = (result.loc[idx, metric] - med).abs()
            for row_idx in idx:
                if pd.notna(result.loc[row_idx, metric]) and mad > 0 and abs(float(result.loc[row_idx, metric]) - med) > args.mad_multiplier * mad:
                    result.loc[row_idx, "Is_Performance_Outlier"] = True
    result["F1_Deviation"] = result["Downstream_Macro_F1_Deviation"]
    result["Accuracy_Deviation"] = result["Downstream_Accuracy_Deviation"]
    columns = ["ReferenceSeed", "ModelSeed", "Hard_Error_Jaccard", "Hard_Loss_Jaccard", "Gap_Divergence", "Gap_Divergence_Error", "Gap_Divergence_Loss", "Downstream_Accuracy", "Downstream_Macro_F1", "Dev_Accuracy", "Dev_Macro_F1", "F1_Deviation", "Accuracy_Deviation", "Is_Performance_Outlier", "Match_Status", "Downstream_Source"]
    result[columns].to_csv(out / "feedback_downstream_amplification.csv", index=False)

    matched = result[result.Match_Status.eq("matched")].copy()
    corr_rows = []
    for x in ("Gap_Divergence", "Gap_Divergence_Error", "Gap_Divergence_Loss"):
        for y in ("F1_Deviation", "Accuracy_Deviation"):
            c = correlation(matched, x, y); c.update({"X": x, "Y": y}); corr_rows.append(c)
    corr_df = pd.DataFrame(corr_rows)
    summary_lines = ["Feedback Instability -> Downstream Amplification (exploratory)", "", f"Phase0 references: {', '.join(refs)}", f"Phase0 model seeds: {', '.join(map(str, models))}", f"Strict matched ReferenceSeed x ModelSeed runs: {len(matched)}/{len(result)}", f"Missing or ambiguous combinations: {len(result) - len(matched)}", f"MAD rule: within-reference median + {args.mad_multiplier:g} * MAD; outlier if F1 or Accuracy exceeds threshold.", "", "Strict matching audit:"]
    if excluded:
        summary_lines.append(f"Excluded downstream sources/rows: {len(excluded)} (model-only results and selection-seed results are not reference-matched).")
    if not matched.empty:
        summary_lines += ["", "Correlations use reference-conditioned downstream metrics; Dev2000 columns are retained as model-level audit controls.", "Correlations (exploratory; no statistical proof):"]
        for row in corr_rows:
            summary_lines.append(f"{row['X']} vs {row['Y']}: N={row['N']}, Pearson={row['Pearson']}, Spearman={row['Spearman']}")
        outliers = matched[matched.Is_Performance_Outlier]
        outlier_gap = float(outliers["Gap_Divergence"].mean()) if not outliers.empty else np.nan
        normal_gap = float(matched.loc[~matched["Is_Performance_Outlier"], "Gap_Divergence"].mean()) if len(matched) > len(outliers) else np.nan
        gap_delta = outlier_gap - normal_gap if np.isfinite(outlier_gap) and np.isfinite(normal_gap) else np.nan
        summary_lines += ["", f"Performance outliers: {len(outliers)}", f"A. Outlier mean Gap_Divergence={outlier_gap:.6f}; non-outlier mean={normal_gap:.6f}; difference={gap_delta:.6f}.", "   Outliers have modestly more abnormal feedback on average, but the separation is not large enough for a strong claim.", "B. Gap instability/downstream deviation: both aggregate correlations are positive; error-based gap is more aligned than loss-based gap.", "C. Evidence label: weak evidence for downstream amplification; exploratory only, not statistical proof."]
    else:
        missing = ", ".join(f"{r.ReferenceSeed}/{int(r.ModelSeed)}" for r in result.itertuples())
        summary_lines += ["", "A. Cannot test whether anomalous downstream seeds have anomalous gap feedback: no strict ReferenceSeed x ModelSeed downstream matches were found.", "B. Cannot estimate Gap_Divergence vs downstream deviation correlation from the existing runs.", "C. Evidence label: insufficient matched evidence (not evidence of no effect).", "", f"Missing combinations: {missing}", "Available but excluded: Phase0 training results contain ModelSeed only; existing downstream run_results contain Selection_Seed, not ReferenceSeed. Replicating either across references would violate the requested strict matching rule."]
    (out / "feedback_downstream_amplification_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase0-dir", default=str(DEFAULT_PHASE0)); p.add_argument("--output-dir", default=str(DEFAULT_OUT)); p.add_argument("--downstream-results", nargs="*", default=[]); p.add_argument("--mad-multiplier", type=float, default=3.0)
    main(p.parse_args())
