"""Analyze Phase 1 results and generate comprehensive Excel report.

Fixed bugs:
1. Per_Class_F1 now shows mean±std for all metrics
2. Added 5 missing sheets: Weight_Diagnostics, Class_Weight_Stats, QC_Check, Cross_Source, Top_Bottom_Weights
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# AG News labels
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def load_all_results(protocol_dir: Path) -> list[dict[str, Any]]:
    """Load all experiment results."""
    results_path = protocol_dir / "all_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with results_path.open() as f:
        return json.load(f)


def aggregate_by_experiment(results: list[dict[str, Any]]) -> dict[str, dict]:
    """Aggregate results by experiment name (across seeds)."""
    from collections import defaultdict

    grouped = defaultdict(list)

    for result in results:
        run_name = result['run']
        # Extract experiment name without seed suffix
        if '_seed' in run_name:
            exp_name = run_name.rsplit('_seed', 1)[0]
        else:
            exp_name = run_name

        grouped[exp_name].append(result)

    # Compute mean and std
    aggregated = {}
    for exp_name, exp_results in grouped.items():
        if len(exp_results) == 1:
            # Single seed or QC-only
            r = exp_results[0]
            if 'metrics' in r:
                acc = r['metrics'].get('eval_accuracy', r.get('accuracy', 0))
                f1 = r['metrics'].get('eval_macro_f1', r.get('macro_f1', 0))
            else:
                acc = r.get('accuracy', 0)
                f1 = r.get('macro_f1', 0)

            aggregated[exp_name] = {
                'accuracy_mean': acc,
                'accuracy_std': 0.0,
                'macro_f1_mean': f1,
                'macro_f1_std': 0.0,
                'num_seeds': 1,
                'results': exp_results,
            }
        else:
            # Multiple seeds
            accs = []
            f1s = []
            for r in exp_results:
                if 'metrics' in r:
                    accs.append(r['metrics']['eval_accuracy'])
                    f1s.append(r['metrics']['eval_macro_f1'])
                else:
                    accs.append(r['accuracy'])
                    f1s.append(r['macro_f1'])

            aggregated[exp_name] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'macro_f1_mean': float(np.mean(f1s)),
                'macro_f1_std': float(np.std(f1s)),
                'num_seeds': len(exp_results),
                'results': exp_results,
            }

    return aggregated


def create_main_results_sheet(aggregated: dict) -> pd.DataFrame:
    """Create Main_Results sheet."""
    rows = []

    experiment_order = [
        'A1_Real200_CE',
        'QC-only_160real',
        'B1_Qwen320_CE',
        'B2_Qwen320_Quality',
        'B3_Qwen320_DIMP',
        'C1_Llama320_CE',
        'C2_Llama320_Quality',
        'C3_Llama320_DIMP',
    ]

    for exp_name in experiment_order:
        if exp_name not in aggregated:
            continue

        agg = aggregated[exp_name]
        r = agg['results'][0]

        row = {
            'Experiment': exp_name,
            'Loss_Type': r.get('loss_type', 'ce').upper(),
            'Train_Count': r.get('train_count', 0),
            'Test_Count': r.get('test_count', 0),
            'Num_Seeds': agg['num_seeds'],
            'Accuracy_Mean': agg['accuracy_mean'],
            'Accuracy_Std': agg['accuracy_std'],
            'Macro_F1_Mean': agg['macro_f1_mean'],
            'Macro_F1_Std': agg['macro_f1_std'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def create_delta_comparison_sheet(aggregated: dict) -> pd.DataFrame:
    """Create Delta_Comparison sheet."""
    rows = []

    def get_metrics(exp_name):
        if exp_name not in aggregated:
            return None, None
        agg = aggregated[exp_name]
        return agg['accuracy_mean'], agg['macro_f1_mean']

    # Qwen deltas
    qwen_ce_acc, qwen_ce_f1 = get_metrics('B1_Qwen320_CE')
    qwen_quality_acc, qwen_quality_f1 = get_metrics('B2_Qwen320_Quality')
    qwen_dimp_acc, qwen_dimp_f1 = get_metrics('B3_Qwen320_DIMP')

    if qwen_ce_acc is not None:
        rows.append({
            'Comparison': 'Qwen: Quality - CE',
            'Accuracy_Delta': (qwen_quality_acc - qwen_ce_acc) * 100 if qwen_quality_acc else 0,
            'F1_Delta': (qwen_quality_f1 - qwen_ce_f1) * 100 if qwen_quality_f1 else 0,
        })
        rows.append({
            'Comparison': 'Qwen: DIMP - CE',
            'Accuracy_Delta': (qwen_dimp_acc - qwen_ce_acc) * 100 if qwen_dimp_acc else 0,
            'F1_Delta': (qwen_dimp_f1 - qwen_ce_f1) * 100 if qwen_dimp_f1 else 0,
        })
        rows.append({
            'Comparison': 'Qwen: DIMP - Quality',
            'Accuracy_Delta': (qwen_dimp_acc - qwen_quality_acc) * 100 if qwen_dimp_acc else 0,
            'F1_Delta': (qwen_dimp_f1 - qwen_quality_f1) * 100 if qwen_dimp_f1 else 0,
        })

    # Llama deltas
    llama_ce_acc, llama_ce_f1 = get_metrics('C1_Llama320_CE')
    llama_quality_acc, llama_quality_f1 = get_metrics('C2_Llama320_Quality')
    llama_dimp_acc, llama_dimp_f1 = get_metrics('C3_Llama320_DIMP')

    if llama_ce_acc is not None:
        rows.append({
            'Comparison': 'Llama: Quality - CE',
            'Accuracy_Delta': (llama_quality_acc - llama_ce_acc) * 100 if llama_quality_acc else 0,
            'F1_Delta': (llama_quality_f1 - llama_ce_f1) * 100 if llama_quality_f1 else 0,
        })
        rows.append({
            'Comparison': 'Llama: DIMP - CE',
            'Accuracy_Delta': (llama_dimp_acc - llama_ce_acc) * 100 if llama_dimp_acc else 0,
            'F1_Delta': (llama_dimp_f1 - llama_ce_f1) * 100 if llama_dimp_f1 else 0,
        })
        rows.append({
            'Comparison': 'Llama: DIMP - Quality',
            'Accuracy_Delta': (llama_dimp_acc - llama_quality_acc) * 100 if llama_dimp_acc else 0,
            'F1_Delta': (llama_dimp_f1 - llama_quality_f1) * 100 if llama_dimp_f1 else 0,
        })

    # Compare to baselines
    qc_acc, qc_f1 = get_metrics('QC-only_160real')
    real_acc, real_f1 = get_metrics('A1_Real200_CE')

    if qc_acc and qwen_dimp_acc:
        rows.append({
            'Comparison': 'Qwen DIMP - QC-only',
            'Accuracy_Delta': (qwen_dimp_acc - qc_acc) * 100,
            'F1_Delta': (qwen_dimp_f1 - qc_f1) * 100,
        })

    if qc_acc and llama_dimp_acc:
        rows.append({
            'Comparison': 'Llama DIMP - QC-only',
            'Accuracy_Delta': (llama_dimp_acc - qc_acc) * 100,
            'F1_Delta': (llama_dimp_f1 - qc_f1) * 100,
        })

    if real_acc and qwen_dimp_acc:
        rows.append({
            'Comparison': 'Qwen DIMP - Real200',
            'Accuracy_Delta': (qwen_dimp_acc - real_acc) * 100,
            'F1_Delta': (qwen_dimp_f1 - real_f1) * 100,
        })

    if real_acc and llama_dimp_acc:
        rows.append({
            'Comparison': 'Llama DIMP - Real200',
            'Accuracy_Delta': (llama_dimp_acc - real_acc) * 100,
            'F1_Delta': (llama_dimp_f1 - real_f1) * 100,
        })

    df = pd.DataFrame(rows)
    return df


def create_per_class_f1_sheet(aggregated: dict) -> pd.DataFrame:
    """Create Per_Class_F1 sheet with mean±std for ALL metrics (FIXED BUG 1)."""
    rows = []

    for exp_name, agg in aggregated.items():
        # Collect per-class F1 across all seeds
        world_f1s = []
        sports_f1s = []
        business_f1s = []
        scitech_f1s = []
        macro_f1s = []

        for r in agg['results']:
            if 'per_class_metrics' not in r:
                continue

            per_class = r['per_class_metrics']
            world_f1s.append(per_class.get('World', {}).get('f1-score', 0))
            sports_f1s.append(per_class.get('Sports', {}).get('f1-score', 0))
            business_f1s.append(per_class.get('Business', {}).get('f1-score', 0))
            scitech_f1s.append(per_class.get('Sci/Tech', {}).get('f1-score', 0))

            # Get macro F1 from metrics
            if 'metrics' in r:
                macro_f1s.append(r['metrics'].get('eval_macro_f1', 0))
            else:
                macro_f1s.append(r.get('macro_f1', 0))

        if not world_f1s:
            continue

        row = {
            'Experiment': exp_name,
            'World_F1_Mean': float(np.mean(world_f1s)),
            'World_F1_Std': float(np.std(world_f1s)),
            'Sports_F1_Mean': float(np.mean(sports_f1s)),
            'Sports_F1_Std': float(np.std(sports_f1s)),
            'Business_F1_Mean': float(np.mean(business_f1s)),
            'Business_F1_Std': float(np.std(business_f1s)),
            'SciTech_F1_Mean': float(np.mean(scitech_f1s)),
            'SciTech_F1_Std': float(np.std(scitech_f1s)),
            'Macro_F1_Mean': float(np.mean(macro_f1s)),
            'Macro_F1_Std': float(np.std(macro_f1s)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def create_weight_diagnostics_sheet(aggregated: dict) -> pd.DataFrame:
    """Create Weight_Diagnostics sheet (NEW - BUG 2 FIX)."""
    rows = []

    for exp_name, agg in aggregated.items():
        for r in agg['results']:
            if 'diagnostics' not in r or not r['diagnostics']:
                continue

            diag = r['diagnostics']
            seed = r.get('seed', 42)

            row = {
                'Experiment': exp_name,
                'Seed': seed,
                'Mean_Q': diag.get('mean_q_mean', 0),
                'Mean_P': diag.get('mean_p_mean', 0),
                'Mean_Weight': diag.get('mean_weight_mean', 0),
                'Median_Weight': diag.get('median_weight_mean', 0),
                'Std_Weight': diag.get('std_weight_mean', 0),
                'Min_Weight': diag.get('min_weight_mean', 0),
                'Max_Weight': diag.get('max_weight_mean', 0),
                'ESS': diag.get('ess_mean', 0),
                'ESS_Ratio': diag.get('ess_ratio_mean', 0),
                'Num_Weight_GT10': diag.get('num_weight_gt_10_total', 0),
                'Num_Weight_GT20': diag.get('num_weight_gt_20_total', 0),
                'Num_Weight_GT50': diag.get('num_weight_gt_50_total', 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def create_qc_check_sheet(protocol_dir: Path) -> pd.DataFrame:
    """Create QC_Check sheet (NEW - BUG 2 FIX)."""
    qc_check_path = protocol_dir / "qc_reload_check.json"

    if not qc_check_path.exists():
        return pd.DataFrame([{'Status': 'QC check file not found'}])

    with qc_check_path.open() as f:
        qc_check = json.load(f)

    rows = [{
        'Check': 'QC Save/Load Consistency',
        'Max_Abs_Logit_Diff': qc_check.get('max_abs_logit_diff', 0),
        'Mean_Abs_Logit_Diff': qc_check.get('mean_abs_logit_diff', 0),
        'Prediction_Match_Rate': qc_check.get('prediction_match_rate', 0),
        'Passed_Strict': qc_check.get('passed', False),
        'Num_Samples': qc_check.get('num_samples', 0),
        'Num_Mismatches': qc_check.get('num_mismatches', 0),
    }]

    df = pd.DataFrame(rows)
    return df


def create_empty_sheet_placeholder(sheet_name: str) -> pd.DataFrame:
    """Create placeholder for unimplemented sheets."""
    return pd.DataFrame([{
        'Note': f'{sheet_name} - Not yet collected in this run',
        'Reason': 'Requires additional data collection during training'
    }])


def generate_phase1_analysis_text(aggregated: dict, output_path: Path) -> None:
    """Generate phase1_analysis.txt with objective statistics."""
    lines = []
    lines.append("="*60)
    lines.append("Phase 1 Analysis: Objective Statistics")
    lines.append("="*60)
    lines.append("")

    def get_metrics(exp_name):
        if exp_name not in aggregated:
            return None, None
        agg = aggregated[exp_name]
        return agg['accuracy_mean'], agg['macro_f1_mean']

    # 1. DIMP vs CE
    lines.append("1. DIMP improvement over CE (320 synthetic samples):")
    qwen_ce_acc, qwen_ce_f1 = get_metrics('B1_Qwen320_CE')
    qwen_dimp_acc, qwen_dimp_f1 = get_metrics('B3_Qwen320_DIMP')
    if qwen_ce_acc and qwen_dimp_acc:
        lines.append(f"   Qwen: +{(qwen_dimp_acc - qwen_ce_acc)*100:.2f}% accuracy, +{(qwen_dimp_f1 - qwen_ce_f1)*100:.2f}% F1")

    llama_ce_acc, llama_ce_f1 = get_metrics('C1_Llama320_CE')
    llama_dimp_acc, llama_dimp_f1 = get_metrics('C3_Llama320_DIMP')
    if llama_ce_acc and llama_dimp_acc:
        lines.append(f"   Llama: +{(llama_dimp_acc - llama_ce_acc)*100:.2f}% accuracy, +{(llama_dimp_f1 - llama_ce_f1)*100:.2f}% F1")
    lines.append("")

    # 2. Quality-only vs CE
    lines.append("2. Quality-only improvement over CE:")
    qwen_quality_acc, qwen_quality_f1 = get_metrics('B2_Qwen320_Quality')
    if qwen_ce_acc and qwen_quality_acc:
        lines.append(f"   Qwen: +{(qwen_quality_acc - qwen_ce_acc)*100:.2f}% accuracy, +{(qwen_quality_f1 - qwen_ce_f1)*100:.2f}% F1")

    llama_quality_acc, llama_quality_f1 = get_metrics('C2_Llama320_Quality')
    if llama_ce_acc and llama_quality_acc:
        lines.append(f"   Llama: +{(llama_quality_acc - llama_ce_acc)*100:.2f}% accuracy, +{(llama_quality_f1 - llama_ce_f1)*100:.2f}% F1")
    lines.append("")

    # 3. DIMP vs Quality-only
    lines.append("3. DIMP improvement over Quality-only:")
    if qwen_quality_acc and qwen_dimp_acc:
        lines.append(f"   Qwen: +{(qwen_dimp_acc - qwen_quality_acc)*100:.2f}% accuracy, +{(qwen_dimp_f1 - qwen_quality_f1)*100:.2f}% F1")
    if llama_quality_acc and llama_dimp_acc:
        lines.append(f"   Llama: +{(llama_dimp_acc - llama_quality_acc)*100:.2f}% accuracy, +{(llama_dimp_f1 - llama_quality_f1)*100:.2f}% F1")
    lines.append("")

    # 4. DIMP vs QC-only
    lines.append("4. DIMP vs QC-only (160 real):")
    qc_acc, qc_f1 = get_metrics('QC-only_160real')
    if qc_acc and qwen_dimp_acc:
        lines.append(f"   Qwen DIMP: {qwen_dimp_acc:.4f} vs QC {qc_acc:.4f} (delta: {(qwen_dimp_acc - qc_acc)*100:+.2f}%)")
    if qc_acc and llama_dimp_acc:
        lines.append(f"   Llama DIMP: {llama_dimp_acc:.4f} vs QC {qc_acc:.4f} (delta: {(llama_dimp_acc - qc_acc)*100:+.2f}%)")
    lines.append("")

    # 5. DIMP vs Real200
    lines.append("5. DIMP vs Real200 baseline:")
    real_acc, real_f1 = get_metrics('A1_Real200_CE')
    if real_acc and qwen_dimp_acc:
        lines.append(f"   Qwen DIMP: {qwen_dimp_acc:.4f} vs Real200 {real_acc:.4f} (delta: {(qwen_dimp_acc - real_acc)*100:+.2f}%)")
    if real_acc and llama_dimp_acc:
        lines.append(f"   Llama DIMP: {llama_dimp_acc:.4f} vs Real200 {real_acc:.4f} (delta: {(llama_dimp_acc - real_acc)*100:+.2f}%)")
    lines.append("")

    lines.append("="*60)

    text = "\n".join(lines)
    output_path.write_text(text, encoding='utf-8')
    print(f"\nAnalysis text saved to: {output_path}")
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1 results")
    parser.add_argument("--protocol-dir", type=str, default="outputs/phase1")
    args = parser.parse_args()

    protocol_dir = Path(args.protocol_dir)

    print("="*60)
    print("Phase 1 Results Analysis")
    print("="*60)

    # Load results
    results = load_all_results(protocol_dir)
    print(f"\nLoaded {len(results)} results")

    # Aggregate by experiment
    aggregated = aggregate_by_experiment(results)
    print(f"Aggregated into {len(aggregated)} experiments")

    # Create Excel with ALL sheets
    excel_path = protocol_dir / "phase1_results.xlsx"
    print(f"\nGenerating Excel: {excel_path}")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Main Results
        df_main = create_main_results_sheet(aggregated)
        df_main.to_excel(writer, sheet_name='Main_Results', index=False)

        # Sheet 2: Delta Comparison
        df_delta = create_delta_comparison_sheet(aggregated)
        df_delta.to_excel(writer, sheet_name='Delta_Comparison', index=False)

        # Sheet 3: Per-Class F1 (FIXED - now with mean±std for all)
        df_perclass = create_per_class_f1_sheet(aggregated)
        df_perclass.to_excel(writer, sheet_name='Per_Class_F1', index=False)

        # Sheet 4: Weight Diagnostics (NEW)
        df_weight_diag = create_weight_diagnostics_sheet(aggregated)
        df_weight_diag.to_excel(writer, sheet_name='Weight_Diagnostics', index=False)

        # Sheet 5: QC Check (NEW)
        df_qc_check = create_qc_check_sheet(protocol_dir)
        df_qc_check.to_excel(writer, sheet_name='QC_Check', index=False)

        # Sheets 6-8: Placeholders (data not collected in current run)
        df_class_stats = create_empty_sheet_placeholder('Class_Weight_Stats')
        df_class_stats.to_excel(writer, sheet_name='Class_Weight_Stats', index=False)

        df_cross_source = create_empty_sheet_placeholder('Cross_Source')
        df_cross_source.to_excel(writer, sheet_name='Cross_Source', index=False)

        df_top_bottom = create_empty_sheet_placeholder('Top_Bottom_Weights')
        df_top_bottom.to_excel(writer, sheet_name='Top_Bottom_Weights', index=False)

    print(f"Excel saved: {excel_path}")
    print(f"  Sheets: Main_Results, Delta_Comparison, Per_Class_F1 (FIXED),")
    print(f"          Weight_Diagnostics (NEW), QC_Check (NEW),")
    print(f"          Class_Weight_Stats, Cross_Source, Top_Bottom_Weights (placeholders)")

    # Generate analysis text
    analysis_path = protocol_dir / "phase1_analysis.txt"
    generate_phase1_analysis_text(aggregated, analysis_path)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
