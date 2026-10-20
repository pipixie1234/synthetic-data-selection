"""Load DIMP diagnostics from Phase 1 experiments.

Loads:
- Per-epoch weight statistics
- Final epoch aggregated statistics
- Per-class weight statistics
- CE and DIMP performance metrics
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_phase1_results(results_path: Path = Path("../dimp/outputs/phase1/all_results.json")) -> list[dict]:
    """Load Phase 1 all_results.json."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with results_path.open() as f:
        return json.load(f)


def extract_final_epoch_diagnostics(result: dict) -> dict[str, Any]:
    """Extract final epoch diagnostics from a single experiment result."""
    if 'diagnostics' not in result or not result['diagnostics']:
        return {}

    diag = result['diagnostics']

    return {
        'mean_weight': diag.get('mean_weight_mean', np.nan),
        'median_weight': diag.get('median_weight_mean', np.nan),
        'std_weight': diag.get('std_weight_mean', np.nan),
        'p90_weight': diag.get('p90_weight_mean', np.nan),
        'p95_weight': diag.get('p95_weight_mean', np.nan),
        'ess': diag.get('ess_mean', np.nan),
        'ess_ratio': diag.get('ess_ratio_mean', np.nan),
        'mean_q': diag.get('mean_q_mean', np.nan),
        'mean_p': diag.get('mean_p_mean', np.nan),
    }


def aggregate_dimp_diagnostics(
    results: list[dict],
    source: str,
    loss_type: str = 'dimp'
) -> dict[str, Any]:
    """Aggregate DIMP diagnostics across seeds for a source.

    Args:
        results: all Phase 1 results
        source: 'qwen' or 'llama'
        loss_type: 'dimp' or 'quality'

    Returns:
        dict with aggregated statistics
    """
    # Filter results for this source and loss type
    filtered = [
        r for r in results
        if source.lower() in r['run'].lower()
        and r.get('loss_type', '').lower() == loss_type.lower()
    ]

    if not filtered:
        return {}

    # Extract diagnostics from each seed
    diagnostics = []
    for r in filtered:
        diag = extract_final_epoch_diagnostics(r)
        if diag:
            diagnostics.append(diag)

    if not diagnostics:
        return {}

    # Aggregate across seeds
    aggregated = {}
    for key in diagnostics[0].keys():
        values = [d[key] for d in diagnostics if not np.isnan(d[key])]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated


def extract_class_level_metrics(results: list[dict]) -> pd.DataFrame:
    """Extract class-level F1 scores for CE and DIMP.

    Returns DataFrame with:
        Source, Class, CE_F1, DIMP_F1, DIMP_Gain_F1
    """
    rows = []

    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    for source in ['qwen', 'llama']:
        # Get CE results
        ce_results = [
            r for r in results
            if source.lower() in r['run'].lower()
            and r.get('loss_type', '').lower() == 'ce'
        ]

        # Get DIMP results
        dimp_results = [
            r for r in results
            if source.lower() in r['run'].lower()
            and r.get('loss_type', '').lower() == 'dimp'
        ]

        if not ce_results or not dimp_results:
            continue

        # Average per-class F1 across seeds
        for label, class_name in label_names.items():
            # CE F1 for this class
            ce_f1s = []
            for r in ce_results:
                if 'per_class_metrics' in r and class_name in r['per_class_metrics']:
                    ce_f1s.append(r['per_class_metrics'][class_name]['f1-score'])

            # DIMP F1 for this class
            dimp_f1s = []
            for r in dimp_results:
                if 'per_class_metrics' in r and class_name in r['per_class_metrics']:
                    dimp_f1s.append(r['per_class_metrics'][class_name]['f1-score'])

            if ce_f1s and dimp_f1s:
                ce_f1_mean = float(np.mean(ce_f1s))
                dimp_f1_mean = float(np.mean(dimp_f1s))
                gain = dimp_f1_mean - ce_f1_mean

                rows.append({
                    'Source': source.capitalize(),
                    'Class': class_name,
                    'CE_F1': ce_f1_mean,
                    'DIMP_F1': dimp_f1_mean,
                    'DIMP_Gain_F1': gain,
                })

    df = pd.DataFrame(rows)
    return df


def extract_overall_metrics(results: list[dict]) -> pd.DataFrame:
    """Extract overall accuracy and F1 for CE and DIMP.

    Returns DataFrame with:
        Source, CE_Accuracy, DIMP_Accuracy, Accuracy_Gain, CE_Macro_F1, DIMP_Macro_F1, Macro_F1_Gain
    """
    rows = []

    for source in ['qwen', 'llama']:
        # CE results
        ce_results = [
            r for r in results
            if source.lower() in r['run'].lower()
            and r.get('loss_type', '').lower() == 'ce'
        ]

        # DIMP results
        dimp_results = [
            r for r in results
            if source.lower() in r['run'].lower()
            and r.get('loss_type', '').lower() == 'dimp'
        ]

        if not ce_results or not dimp_results:
            continue

        # Aggregate CE metrics
        ce_accs = [r['metrics']['eval_accuracy'] for r in ce_results if 'metrics' in r]
        ce_f1s = [r['metrics']['eval_macro_f1'] for r in ce_results if 'metrics' in r]

        # Aggregate DIMP metrics
        dimp_accs = [r['metrics']['eval_accuracy'] for r in dimp_results if 'metrics' in r]
        dimp_f1s = [r['metrics']['eval_macro_f1'] for r in dimp_results if 'metrics' in r]

        if ce_accs and dimp_accs:
            ce_acc = float(np.mean(ce_accs))
            dimp_acc = float(np.mean(dimp_accs))
            ce_f1 = float(np.mean(ce_f1s))
            dimp_f1 = float(np.mean(dimp_f1s))

            rows.append({
                'Source': source.capitalize(),
                'CE_Accuracy': ce_acc,
                'DIMP_Accuracy': dimp_acc,
                'Accuracy_Gain': dimp_acc - ce_acc,
                'CE_Macro_F1': ce_f1,
                'DIMP_Macro_F1': dimp_f1,
                'Macro_F1_Gain': dimp_f1 - ce_f1,
            })

    df = pd.DataFrame(rows)
    return df


def load_dimp_diagnostics() -> dict[str, Any]:
    """Load all DIMP diagnostics from Phase 1.

    Returns:
        dict with overall and class-level diagnostics
    """
    print("="*60)
    print("Loading DIMP Diagnostics from Phase 1")
    print("="*60)

    results = load_phase1_results()
    print(f"Loaded {len(results)} experiment results")

    # Overall diagnostics
    print("\nAggregating overall DIMP diagnostics...")
    qwen_dimp = aggregate_dimp_diagnostics(results, 'qwen', 'dimp')
    llama_dimp = aggregate_dimp_diagnostics(results, 'llama', 'dimp')

    # Class-level metrics
    print("Extracting class-level F1 scores...")
    class_metrics = extract_class_level_metrics(results)
    print(f"  Extracted {len(class_metrics)} class-level rows")

    # Overall metrics
    print("Extracting overall metrics...")
    overall_metrics = extract_overall_metrics(results)
    print(f"  Extracted {len(overall_metrics)} overall rows")

    return {
        'qwen_dimp_overall': qwen_dimp,
        'llama_dimp_overall': llama_dimp,
        'class_level_metrics': class_metrics,
        'overall_metrics': overall_metrics,
    }


if __name__ == "__main__":
    diagnostics = load_dimp_diagnostics()

    print("\n" + "="*60)
    print("DIMP Diagnostics Summary")
    print("="*60)

    print("\nQwen DIMP overall:")
    for key, value in diagnostics['qwen_dimp_overall'].items():
        print(f"  {key}: {value:.4f}")

    print("\nLlama DIMP overall:")
    for key, value in diagnostics['llama_dimp_overall'].items():
        print(f"  {key}: {value:.4f}")

    print("\nClass-level metrics:")
    print(diagnostics['class_level_metrics'])

    print("\nOverall metrics:")
    print(diagnostics['overall_metrics'])
