"""Analyze DIMP training results and generate comparison tables.

This script loads the results from DIMP experiments and generates
analysis reports, comparison tables, and visualizations.

Usage:
  python analyze_dimp_results.py \
    --results-dir outputs/dimp_experiment/dimp_results

  # With baseline comparison
  python analyze_dimp_results.py \
    --results-dir outputs/dimp_experiment/dimp_results \
    --baseline-results outputs/comparison_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_dimp_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load DIMP experiment results."""
    all_results_path = results_dir / "all_results.json"

    if not all_results_path.exists():
        raise FileNotFoundError(f"Results file not found: {all_results_path}")

    with all_results_path.open() as f:
        return json.load(f)


def load_baseline_results(baseline_path: Path) -> list[dict[str, Any]]:
    """Load baseline CE results for comparison."""
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    with baseline_path.open() as f:
        return json.load(f)


def print_results_table(results: list[dict[str, Any]], title: str = "Results") -> None:
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    print(f"{'Experiment':<40} {'Accuracy':<12} {'Macro F1':<12} {'Train':<8} {'Test':<8}")
    print("-" * 80)

    for r in results:
        exp_name = r['run']
        acc = r['metrics']['eval_accuracy']
        f1 = r['metrics']['eval_macro_f1']
        train_count = r.get('train_count', 'N/A')
        test_count = r.get('test_count', 'N/A')

        print(f"{exp_name:<40} {acc:<12.4f} {f1:<12.4f} {train_count:<8} {test_count:<8}")


def print_diagnostics_table(results: list[dict[str, Any]]) -> None:
    """Print DIMP weight diagnostics table."""
    print(f"\n{'='*80}")
    print("DIMP Weight Diagnostics")
    print(f"{'='*80}")

    has_diagnostics = any('diagnostics' in r and r['diagnostics'] for r in results)

    if not has_diagnostics:
        print("No diagnostics available (standard CE loss was used)")
        return

    print(f"{'Experiment':<30} {'Mean W':<10} {'Std W':<10} {'Min W':<10} {'Max W':<10} {'Q_i':<10}")
    print("-" * 80)

    for r in results:
        if 'diagnostics' not in r or not r['diagnostics']:
            continue

        diag = r['diagnostics']
        exp_name = r['run'][:29]

        mean_w = diag.get('mean_weight', 0)
        std_w = diag.get('std_weight', 0)
        min_w = diag.get('min_weight', 0)
        max_w = diag.get('max_weight', 0)
        mean_q = diag.get('mean_quality_score', 0)

        print(f"{exp_name:<30} {mean_w:<10.4f} {std_w:<10.4f} {min_w:<10.4f} "
              f"{max_w:<10.4f} {mean_q:<10.4f}")


def compare_with_baseline(
    dimp_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
) -> None:
    """Compare DIMP results with baseline CE results."""
    print(f"\n{'='*80}")
    print("DIMP vs Baseline CE Comparison")
    print(f"{'='*80}")

    # Map experiments (you may need to adjust mapping based on your baseline naming)
    experiment_map = {
        'exp1_qwen_dimp_test_real': 'exp2_qwen50_test_real',
        'exp2_llama_dimp_test_real': 'exp3_gpt50_test_real',  # Assuming GPT baseline
    }

    print(f"{'Setup':<25} {'Method':<10} {'Accuracy':<12} {'Macro F1':<12} {'Δ Acc':<10} {'Δ F1':<10}")
    print("-" * 85)

    for dimp_exp in dimp_results:
        dimp_name = dimp_exp['run']
        baseline_name = experiment_map.get(dimp_name)

        if baseline_name is None:
            continue

        # Find baseline
        baseline_exp = next((b for b in baseline_results if baseline_name in b.get('run', '')), None)

        if baseline_exp is None:
            print(f"{dimp_name[:24]:<25} {'DIMP':<10} {dimp_exp['metrics']['eval_accuracy']:<12.4f} "
                  f"{dimp_exp['metrics']['eval_macro_f1']:<12.4f} {'N/A':<10} {'N/A':<10}")
            continue

        # Compare
        dimp_acc = dimp_exp['metrics']['eval_accuracy']
        dimp_f1 = dimp_exp['metrics']['eval_macro_f1']
        baseline_acc = baseline_exp['metrics']['eval_accuracy']
        baseline_f1 = baseline_exp['metrics']['eval_macro_f1']

        delta_acc = dimp_acc - baseline_acc
        delta_f1 = dimp_f1 - baseline_f1

        setup = dimp_name.replace('_dimp', '').replace('exp1_', '').replace('exp2_', '')[:24]

        print(f"{setup:<25} {'Baseline':<10} {baseline_acc:<12.4f} {baseline_f1:<12.4f} {'-':<10} {'-':<10}")
        print(f"{'':<25} {'DIMP':<10} {dimp_acc:<12.4f} {dimp_f1:<12.4f} "
              f"{delta_acc:+10.4f} {delta_f1:+10.4f}")
        print()


def analyze_cross_generator_transfer(results: list[dict[str, Any]]) -> None:
    """Analyze cross-generator transfer results."""
    print(f"\n{'='*80}")
    print("Cross-Generator Transfer Analysis")
    print(f"{'='*80}")

    # Find experiments
    qwen_to_llama = next((r for r in results if 'qwen_dimp_test_llama' in r['run']), None)
    llama_to_qwen = next((r for r in results if 'llama_dimp_test_qwen' in r['run']), None)

    if qwen_to_llama is None or llama_to_qwen is None:
        print("Cross-generator experiments not found")
        return

    print(f"{'Transfer Direction':<30} {'Accuracy':<12} {'Macro F1':<12}")
    print("-" * 55)

    qwen_llama_acc = qwen_to_llama['metrics']['eval_accuracy']
    qwen_llama_f1 = qwen_to_llama['metrics']['eval_macro_f1']
    print(f"{'QWEN → Llama':<30} {qwen_llama_acc:<12.4f} {qwen_llama_f1:<12.4f}")

    llama_qwen_acc = llama_to_qwen['metrics']['eval_accuracy']
    llama_qwen_f1 = llama_to_qwen['metrics']['eval_macro_f1']
    print(f"{'Llama → QWEN':<30} {llama_qwen_acc:<12.4f} {llama_qwen_f1:<12.4f}")

    # Average
    avg_acc = (qwen_llama_acc + llama_qwen_acc) / 2
    avg_f1 = (qwen_llama_f1 + llama_qwen_f1) / 2
    print("-" * 55)
    print(f"{'Average Cross-Transfer':<30} {avg_acc:<12.4f} {avg_f1:<12.4f}")


def generate_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    """Generate summary report."""
    summary = {
        "total_experiments": len(results),
        "experiments": {},
        "best_accuracy": {"run": None, "accuracy": 0},
        "best_f1": {"run": None, "f1": 0},
    }

    for r in results:
        run_name = r['run']
        acc = r['metrics']['eval_accuracy']
        f1 = r['metrics']['eval_macro_f1']

        summary['experiments'][run_name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "train_count": r.get('train_count', 0),
            "test_count": r.get('test_count', 0),
            "diagnostics": r.get('diagnostics', {}),
        }

        if acc > summary['best_accuracy']['accuracy']:
            summary['best_accuracy'] = {"run": run_name, "accuracy": acc}

        if f1 > summary['best_f1']['f1']:
            summary['best_f1'] = {"run": run_name, "f1": f1}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DIMP training results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing DIMP results",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Path to baseline CE results JSON (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save summary JSON (optional)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load DIMP results
    print("Loading DIMP results...")
    dimp_results = load_dimp_results(results_dir)

    # Print main results table
    print_results_table(dimp_results, "DIMP Experiment Results")

    # Print diagnostics
    print_diagnostics_table(dimp_results)

    # Cross-generator transfer
    analyze_cross_generator_transfer(dimp_results)

    # Compare with baseline if available
    if args.baseline_results:
        try:
            baseline_results = load_baseline_results(Path(args.baseline_results))
            compare_with_baseline(dimp_results, baseline_results)
        except Exception as e:
            print(f"\nWarning: Could not load baseline results: {e}")

    # Generate summary
    if args.output:
        generate_summary(dimp_results, Path(args.output))
    else:
        default_output = results_dir / "analysis_summary.json"
        generate_summary(dimp_results, default_output)

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
