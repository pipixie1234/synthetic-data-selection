"""Main analysis script for distribution diagnostics.

Combines:
1. Word overlap analysis
2. MMD analysis
3. DIMP diagnostics
4. Correlation analysis
5. Excel generation
"""
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from compute_embeddings import compute_and_cache_all_embeddings
from compute_mmd import compute_all_mmd_matrices, create_mmd_dataframes
from load_dimp_diagnostics import load_dimp_diagnostics
from load_phase1_data import load_all_data
from word_overlap import analyze_word_overlap


def create_synthetic_vs_real_table(mmd_results: dict) -> pd.DataFrame:
    """Create Synthetic_vs_Real summary table.

    Columns: Source, Scope, MMD_to_Real, MMD_std
    """
    rows = []

    scopes = ['overall', 'world', 'sports', 'business', 'sci_tech']
    scope_names = {
        'overall': 'Overall',
        'world': 'World',
        'sports': 'Sports',
        'business': 'Business',
        'sci_tech': 'SciTech',
    }

    for scope in scopes:
        if scope not in mmd_results:
            continue

        details = mmd_results[scope]['details']

        # Qwen vs Real
        if 'real_vs_qwen' in details:
            stats = details['real_vs_qwen']
            rows.append({
                'Source': 'Qwen',
                'Scope': scope_names[scope],
                'MMD_to_Real': stats['mean'],
                'MMD_std': stats['std'],
            })

        # Llama vs Real
        if 'real_vs_llama' in details:
            stats = details['real_vs_llama']
            rows.append({
                'Source': 'Llama',
                'Scope': scope_names[scope],
                'MMD_to_Real': stats['mean'],
                'MMD_std': stats['std'],
            })

    df = pd.DataFrame(rows)
    return df


def merge_mmd_with_dimp(
    synthetic_vs_real: pd.DataFrame,
    dimp_diagnostics: dict
) -> pd.DataFrame:
    """Merge MMD with DIMP diagnostics.

    Returns DataFrame with:
        Source, Scope, MMD_to_Real, Mean_DIMP_Weight, Median_DIMP_Weight,
        P95_DIMP_Weight, ESS_Ratio, CE_F1, DIMP_F1, DIMP_Gain_F1
    """
    # Class-level metrics
    class_metrics = dimp_diagnostics['class_level_metrics']

    # Overall metrics
    overall_metrics = dimp_diagnostics['overall_metrics']

    rows = []

    for _, row in synthetic_vs_real.iterrows():
        source = row['Source']
        scope = row['Scope']

        new_row = {
            'Source': source,
            'Scope': scope,
            'MMD_to_Real': row['MMD_to_Real'],
            'MMD_std': row['MMD_std'],
        }

        # Add DIMP weight statistics
        if scope == 'Overall':
            # Overall DIMP statistics
            dimp_key = f"{source.lower()}_dimp_overall"
            if dimp_key in dimp_diagnostics:
                dimp_stats = dimp_diagnostics[dimp_key]
                new_row['Mean_DIMP_Weight'] = dimp_stats.get('mean_weight_mean', np.nan)
                new_row['Median_DIMP_Weight'] = dimp_stats.get('median_weight_mean', np.nan)
                new_row['P95_DIMP_Weight'] = dimp_stats.get('p95_weight_mean', np.nan)
                new_row['ESS_Ratio'] = dimp_stats.get('ess_ratio_mean', np.nan)

            # Overall performance
            overall_row = overall_metrics[overall_metrics['Source'] == source]
            if not overall_row.empty:
                new_row['CE_Accuracy'] = overall_row.iloc[0]['CE_Accuracy']
                new_row['DIMP_Accuracy'] = overall_row.iloc[0]['DIMP_Accuracy']
                new_row['Accuracy_Gain'] = overall_row.iloc[0]['Accuracy_Gain']
                new_row['CE_F1'] = overall_row.iloc[0]['CE_Macro_F1']
                new_row['DIMP_F1'] = overall_row.iloc[0]['DIMP_Macro_F1']
                new_row['DIMP_Gain_F1'] = overall_row.iloc[0]['Macro_F1_Gain']
        else:
            # Class-level statistics
            # DIMP diagnostics use the display name "Sci/Tech", while the
            # MMD summary uses "SciTech". Normalize this one label when
            # joining the two tables.
            dimp_scope = 'Sci/Tech' if scope == 'SciTech' else scope
            class_row = class_metrics[
                (class_metrics['Source'] == source) &
                (class_metrics['Class'] == dimp_scope)
            ]

            if not class_row.empty:
                new_row['CE_F1'] = class_row.iloc[0]['CE_F1']
                new_row['DIMP_F1'] = class_row.iloc[0]['DIMP_F1']
                new_row['DIMP_Gain_F1'] = class_row.iloc[0]['DIMP_Gain_F1']

            # Note: class-level weight statistics not available in current diagnostics
            new_row['Mean_DIMP_Weight'] = np.nan
            new_row['Median_DIMP_Weight'] = np.nan
            new_row['P95_DIMP_Weight'] = np.nan
            new_row['ESS_Ratio'] = np.nan

        rows.append(new_row)

    df = pd.DataFrame(rows)
    return df


def compute_correlation_analysis(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between MMD and DIMP metrics.

    Only uses class-level data (8 points: 2 sources × 4 classes).
    """
    # Filter to class-level only (exclude Overall)
    class_df = merged_df[merged_df['Scope'] != 'Overall'].copy()

    if len(class_df) < 3:
        print("Warning: Not enough class-level data for correlation analysis")
        return pd.DataFrame()

    # Remove rows with NaN in key columns
    class_df = class_df.dropna(subset=['MMD_to_Real', 'DIMP_Gain_F1'])

    if len(class_df) < 3:
        print("Warning: Not enough valid data after removing NaNs")
        return pd.DataFrame()

    X = class_df['MMD_to_Real'].values
    Y_gain = class_df['DIMP_Gain_F1'].values

    rows = []

    # MMD vs DIMP Gain F1
    if len(X) >= 3:
        pearson_r, pearson_p = pearsonr(X, Y_gain)
        spearman_rho, spearman_p = spearmanr(X, Y_gain)

        rows.append({
            'Metric_X': 'MMD_to_Real',
            'Metric_Y': 'DIMP_Gain_F1',
            'N': len(X),
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_rho': spearman_rho,
            'Spearman_p': spearman_p,
        })

    df = pd.DataFrame(rows)
    return df


def create_figures(merged_df: pd.DataFrame, mmd_dfs: dict, output_dir: Path):
    """Generate visualization plots."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Filter class-level data for scatter plots
    class_df = merged_df[merged_df['Scope'] != 'Overall'].dropna(subset=['MMD_to_Real', 'DIMP_Gain_F1'])

    if len(class_df) < 3:
        print("Warning: Not enough data for scatter plots")
        return

    # Plot 1: MMD vs DIMP Gain F1
    plt.figure(figsize=(8, 6))
    for source in ['Qwen', 'Llama']:
        source_df = class_df[class_df['Source'] == source]
        plt.scatter(
            source_df['MMD_to_Real'],
            source_df['DIMP_Gain_F1'],
            label=source,
            s=100,
            alpha=0.7
        )

        # Annotate points
        for _, row in source_df.iterrows():
            label = f"{row['Source']}-{row['Scope']}"
            plt.annotate(label, (row['MMD_to_Real'], row['DIMP_Gain_F1']),
                        fontsize=8, alpha=0.7)

    plt.xlabel('MMD to Real (semantic distribution discrepancy)')
    plt.ylabel('DIMP Gain F1 (DIMP_F1 - CE_F1)')
    plt.title('MMD vs DIMP Performance Gain (Class-level)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "mmd_vs_dimp_gain.png", dpi=150)
    plt.close()

    # Plot 2: MMD heatmap (Overall)
    if 'overall' in mmd_dfs:
        plt.figure(figsize=(6, 5))
        df = mmd_dfs['overall']
        im = plt.imshow(df.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='MMD²')
        plt.xticks(range(3), df.columns)
        plt.yticks(range(3), df.index)
        plt.title('Overall MMD Matrix')

        # Annotate values
        for i in range(3):
            for j in range(3):
                text = plt.text(j, i, f'{df.values[i, j]:.4f}',
                              ha="center", va="center", color="black", fontsize=10)

        plt.tight_layout()
        plt.savefig(fig_dir / "mmd_heatmap_overall.png", dpi=150)
        plt.close()

    print(f"\nSaved figures to {fig_dir}/")


def generate_analysis_text(
    word_overlap_results: dict,
    mmd_results: dict,
    merged_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    output_path: Path
):
    """Generate distribution_analysis.txt with answers to key questions."""
    lines = []
    lines.append("="*60)
    lines.append("Distribution Analysis Report")
    lines.append("="*60)
    lines.append("")

    # 1. Overall MMD comparison
    lines.append("1. Qwen vs Llama: Overall MMD to Real")
    overall_details = mmd_results['overall']['details']
    qwen_mmd = overall_details['real_vs_qwen']['mean']
    llama_mmd = overall_details['real_vs_llama']['mean']

    if qwen_mmd > llama_mmd:
        lines.append(f"   Qwen has LARGER MMD to Real: {qwen_mmd:.6f} vs {llama_mmd:.6f}")
    else:
        lines.append(f"   Llama has LARGER MMD to Real: {llama_mmd:.6f} vs {qwen_mmd:.6f}")
    lines.append("")

    # 2. Which class has largest MMD
    lines.append("2. Which class has the largest synthetic-real MMD?")
    class_df = merged_df[merged_df['Scope'] != 'Overall']
    if not class_df.empty:
        max_row = class_df.loc[class_df['MMD_to_Real'].idxmax()]
        lines.append(f"   {max_row['Source']}-{max_row['Scope']}: MMD = {max_row['MMD_to_Real']:.6f}")
    lines.append("")

    # 3. Word overlap vs MMD consistency
    lines.append("3. Word overlap vs MMD consistency:")
    raw_df = word_overlap_results['df_raw']
    qwen_word_overlap = raw_df.loc['Qwen', 'Real']
    llama_word_overlap = raw_df.loc['Llama', 'Real']

    lines.append(f"   Word overlap: Qwen={qwen_word_overlap:.4f}, Llama={llama_word_overlap:.4f}")
    lines.append(f"   MMD to Real: Qwen={qwen_mmd:.6f}, Llama={llama_mmd:.6f}")

    word_winner = "Qwen" if qwen_word_overlap > llama_word_overlap else "Llama"
    mmd_winner = "Llama" if qwen_mmd > llama_mmd else "Qwen"  # Lower MMD = more similar

    if word_winner == mmd_winner:
        lines.append(f"   CONSISTENT: Both metrics favor {word_winner}")
    else:
        lines.append(f"   INCONSISTENT: Word overlap favors {word_winner}, MMD favors {mmd_winner}")
    lines.append("")

    # 4. Correlation analysis
    lines.append("4. MMD vs DIMP Gain correlation (class-level, N=8):")
    if not correlation_df.empty:
        row = correlation_df.iloc[0]
        lines.append(f"   Pearson r = {row['Pearson_r']:.4f}, p = {row['Pearson_p']:.4f}")
        lines.append(f"   Spearman rho = {row['Spearman_rho']:.4f}, p = {row['Spearman_p']:.4f}")

        if abs(row['Pearson_r']) > 0.3:
            direction = "POSITIVE" if row['Pearson_r'] > 0 else "NEGATIVE"
            lines.append(f"   Trend: {direction} correlation observed")
        else:
            lines.append("   Trend: WEAK or NO clear correlation")
    else:
        lines.append("   Not enough data for correlation analysis")
    lines.append("")

    # 5. Exploratory hypothesis
    lines.append("5. Exploratory hypothesis:")
    lines.append('   "Larger MMD → larger DIMP gain"')
    if not correlation_df.empty:
        row = correlation_df.iloc[0]
        if row['Pearson_r'] > 0 and row['Pearson_p'] < 0.1:
            lines.append("   WEAKLY SUPPORTED: Positive trend observed (N=8, exploratory)")
        elif row['Pearson_r'] > 0:
            lines.append("   NOT SUPPORTED: Positive direction but not significant")
        else:
            lines.append("   NOT SUPPORTED: Negative or no correlation")
    else:
        lines.append("   CANNOT EVALUATE: Insufficient data")
    lines.append("")

    lines.append("="*60)
    lines.append("Note: All correlation analyses are exploratory (N=8).")
    lines.append("="*60)

    text = "\n".join(lines)
    output_path.write_text(text, encoding='utf-8')
    print(f"\nAnalysis text saved to: {output_path}")
    print(text)


def generate_excel_report(
    word_overlap_results: dict,
    mmd_results: dict,
    mmd_dfs: dict,
    dimp_diagnostics: dict,
    synthetic_vs_real: pd.DataFrame,
    merged_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    output_path: Path
):
    """Generate comprehensive Excel report."""
    print(f"\nGenerating Excel report: {output_path}")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Word Overlap
        ws_name = 'Word_Overlap'
        startrow = 0

        # Raw overlap
        word_overlap_results['df_raw'].to_excel(writer, sheet_name=ws_name, startrow=startrow)
        startrow += len(word_overlap_results['df_raw']) + 3

        # Write label
        label_df = pd.DataFrame([['Content Word Overlap (without stopwords):']])
        label_df.to_excel(writer, sheet_name=ws_name, startrow=startrow, header=False, index=False)
        startrow += 2

        # Content overlap
        word_overlap_results['df_content'].to_excel(writer, sheet_name=ws_name, startrow=startrow)

        # Sheet 2-6: MMD matrices with details
        for scope in ['overall', 'world', 'sports', 'business', 'sci_tech']:
            if scope not in mmd_results:
                continue

            sheet_name = f"MMD_{scope.capitalize()}"
            startrow = 0

            # MMD matrix
            mmd_dfs[scope].to_excel(writer, sheet_name=sheet_name, startrow=startrow)
            startrow += len(mmd_dfs[scope]) + 3

            # Details table
            details = mmd_results[scope]['details']
            detail_rows = []
            for pair, stats in details.items():
                detail_rows.append({
                    'Pair': pair,
                    'MMD_mean': stats['mean'],
                    'MMD_std': stats['std'],
                    'MMD_min': stats['min'],
                    'MMD_p25': stats['p25'],
                    'MMD_median': stats['median'],
                    'MMD_p75': stats['p75'],
                    'MMD_max': stats['max'],
                    'Sigma_mean': stats['sigma_mean'],
                    'Sigma_std': stats['sigma_std'],
                })

            detail_df = pd.DataFrame(detail_rows)
            detail_df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)

        # Sheet 7: MMD_DIMP_Relationship
        merged_df.to_excel(writer, sheet_name='MMD_DIMP_Relationship', index=False)

        # Sheet 8: Correlation_Analysis
        if not correlation_df.empty:
            correlation_df.to_excel(writer, sheet_name='Correlation_Analysis', index=False)

        # Sheet 9: Overall_Summary
        overall_summary = dimp_diagnostics['overall_metrics'].copy()
        # Add MMD
        for _, row in overall_summary.iterrows():
            source = row['Source']
            mmd_row = synthetic_vs_real[
                (synthetic_vs_real['Source'] == source) &
                (synthetic_vs_real['Scope'] == 'Overall')
            ]
            if not mmd_row.empty:
                overall_summary.loc[overall_summary['Source'] == source, 'Overall_MMD_to_Real'] = mmd_row.iloc[0]['MMD_to_Real']

        overall_summary.to_excel(writer, sheet_name='Overall_Summary', index=False)

        # Sheet 10: Synthetic_vs_Real
        synthetic_vs_real.to_excel(writer, sheet_name='Synthetic_vs_Real', index=False)

    print(f"Excel saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Distribution analysis")
    parser.add_argument("--output-dir", type=str, default="outputs/distribution_analysis")
    parser.add_argument("--mmd-sample-size", type=int, default=200,
                       help="Sample size for overall MMD")
    parser.add_argument("--mmd-sample-size-class", type=int, default=50,
                       help="Sample size for class-level MMD")
    parser.add_argument("--mmd-repeats", type=int, default=100,
                       help="Number of bootstrap repeats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute", action='store_true')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Distribution Analysis Pipeline")
    print("="*60)

    # Step 1: Load data
    print("\nStep 1: Loading Phase 1 data...")
    all_data = load_all_data()

    # Step 2: Word overlap
    print("\nStep 2: Word overlap analysis...")
    word_overlap_results = analyze_word_overlap(all_data)

    # Save word overlap CSVs
    word_overlap_results['df_raw'].to_csv(output_dir / "word_overlap_raw.csv")
    word_overlap_results['df_content'].to_csv(output_dir / "word_overlap_content.csv")

    # Step 3: Compute embeddings
    print("\nStep 3: Computing semantic embeddings...")
    embeddings_data = compute_and_cache_all_embeddings(
        all_data,
        cache_dir=output_dir / "embeddings",
        force_recompute=args.force_recompute
    )

    # Step 4: Compute MMD
    print("\nStep 4: Computing MMD matrices...")
    mmd_results = compute_all_mmd_matrices(
        embeddings_data,
        sample_size_overall=args.mmd_sample_size,
        sample_size_class=args.mmd_sample_size_class,
        num_repeats=args.mmd_repeats,
        seed=args.seed
    )

    mmd_dfs = create_mmd_dataframes(mmd_results)

    # Save MMD CSVs
    for scope, df in mmd_dfs.items():
        df.to_csv(output_dir / f"mmd_{scope}.csv")

    # Step 5: Load DIMP diagnostics
    print("\nStep 5: Loading DIMP diagnostics...")
    dimp_diagnostics = load_dimp_diagnostics()

    # Step 6: Create synthetic vs real table
    print("\nStep 6: Creating Synthetic_vs_Real table...")
    synthetic_vs_real = create_synthetic_vs_real_table(mmd_results)
    synthetic_vs_real.to_csv(output_dir / "synthetic_vs_real_mmd.csv", index=False)

    # Step 7: Merge MMD with DIMP
    print("\nStep 7: Merging MMD with DIMP diagnostics...")
    merged_df = merge_mmd_with_dimp(synthetic_vs_real, dimp_diagnostics)
    merged_df.to_csv(output_dir / "mmd_dimp_relationship.csv", index=False)

    # Step 8: Correlation analysis
    print("\nStep 8: Computing correlations...")
    correlation_df = compute_correlation_analysis(merged_df)
    if not correlation_df.empty:
        correlation_df.to_csv(output_dir / "mmd_correlation_results.csv", index=False)

    # Step 9: Generate figures
    print("\nStep 9: Generating figures...")
    create_figures(merged_df, mmd_dfs, output_dir)

    # Step 10: Generate Excel
    print("\nStep 10: Generating Excel report...")
    excel_path = output_dir / "distribution_analysis.xlsx"
    generate_excel_report(
        word_overlap_results,
        mmd_results,
        mmd_dfs,
        dimp_diagnostics,
        synthetic_vs_real,
        merged_df,
        correlation_df,
        excel_path
    )

    # Step 11: Generate analysis text
    print("\nStep 11: Generating analysis text...")
    text_path = output_dir / "distribution_analysis.txt"
    generate_analysis_text(
        word_overlap_results,
        mmd_results,
        merged_df,
        correlation_df,
        text_path
    )

    print("\n" + "="*60)
    print("Distribution Analysis Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}/")


if __name__ == "__main__":
    main()
