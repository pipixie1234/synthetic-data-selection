#!/bin/bash
# Complete distribution analysis pipeline
# Run this directly in terminal for faster execution

set -e  # Exit on error

echo "============================================================"
echo "Distribution Analysis Pipeline"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Encoder: bert-base-uncased (cached)"
echo "  - MMD sample size: 160 (overall), 40 (per class)"
echo "  - MMD repeats: 100"
echo "  - Data: Phase 1 DIMP fixed splits (Real:160, Qwen:320, Llama:320)"
echo ""

# Create output directory
mkdir -p outputs/distribution_analysis/embeddings
mkdir -p outputs/distribution_analysis/figures

# Step 1: Word Overlap
echo "Step 1/5: Computing word overlap..."
python3 -c "
from load_phase1_data import load_all_data
from word_overlap import analyze_word_overlap
from pathlib import Path

output_dir = Path('outputs/distribution_analysis')
all_data = load_all_data()
word_overlap_results = analyze_word_overlap(all_data)
word_overlap_results['df_raw'].to_csv(output_dir / 'word_overlap_raw.csv')
word_overlap_results['df_content'].to_csv(output_dir / 'word_overlap_content.csv')
print('✓ Word overlap saved')
"

# Step 2: Compute Embeddings
echo ""
echo "Step 2/5: Computing semantic embeddings (this may take a few minutes)..."
python3 -c "
from load_phase1_data import load_all_data
from compute_embeddings import compute_and_cache_all_embeddings
from pathlib import Path

output_dir = Path('outputs/distribution_analysis')
all_data = load_all_data()
embeddings_data = compute_and_cache_all_embeddings(
    all_data,
    cache_dir=output_dir / 'embeddings',
    force_recompute=False
)
print('✓ Embeddings cached')
"

# Step 3: Compute MMD
echo ""
echo "Step 3/5: Computing MMD matrices..."
python3 -c "
from load_phase1_data import load_all_data
from compute_embeddings import compute_and_cache_all_embeddings
from compute_mmd import compute_all_mmd_matrices, create_mmd_dataframes
from pathlib import Path

output_dir = Path('outputs/distribution_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

all_data = load_all_data()
embeddings_data = compute_and_cache_all_embeddings(
    all_data,
    cache_dir=output_dir / 'embeddings',
    force_recompute=False
)

mmd_results = compute_all_mmd_matrices(
    embeddings_data,
    sample_size_overall=160,
    sample_size_class=40,
    num_repeats=100,
    seed=42
)

mmd_dfs = create_mmd_dataframes(mmd_results)
for scope, df in mmd_dfs.items():
    csv_path = str(output_dir / f'mmd_{scope}.csv')
    df.to_csv(csv_path)

print('✓ MMD matrices saved')
"

# Step 4: Load DIMP Diagnostics
echo ""
echo "Step 4/5: Loading DIMP diagnostics..."
python3 -c "
from load_dimp_diagnostics import load_dimp_diagnostics
from pathlib import Path

output_dir = Path('outputs/distribution_analysis')
diagnostics = load_dimp_diagnostics()
diagnostics['class_level_metrics'].to_csv(output_dir / 'dimp_class_metrics.csv', index=False)
diagnostics['overall_metrics'].to_csv(output_dir / 'dimp_overall_metrics.csv', index=False)
print('✓ DIMP diagnostics saved')
"

# Step 5: Generate Full Report
echo ""
echo "Step 5/5: Generating comprehensive report..."
python3 << 'EOF'
from analyze_distribution import (
    create_synthetic_vs_real_table,
    merge_mmd_with_dimp,
    compute_correlation_analysis,
    create_figures,
    generate_analysis_text,
    generate_excel_report
)
from load_dimp_diagnostics import load_dimp_diagnostics
import pandas as pd
from pathlib import Path

output_dir = Path('outputs/distribution_analysis')

# Load pre-computed results
word_overlap_results = {
    'df_raw': pd.read_csv(output_dir / 'word_overlap_raw.csv', index_col=0),
    'df_content': pd.read_csv(output_dir / 'word_overlap_content.csv', index_col=0)
}

mmd_dfs = {}
for scope in ['overall', 'world', 'sports', 'business', 'sci_tech']:
    csv_path = output_dir / f'mmd_{scope}.csv'
    if csv_path.exists():
        mmd_dfs[scope] = pd.read_csv(csv_path, index_col=0)

# Create mmd_results structure
mmd_results = {}
for scope, df in mmd_dfs.items():
    mmd_results[scope] = {
        'matrix': df.values,
        'details': {
            'real_vs_qwen': {'mean': df.loc['Real', 'Qwen'], 'std': 0.0, 'min': 0, 'p25': 0, 'median': 0, 'p75': 0, 'max': 0, 'sigma_mean': 0, 'sigma_std': 0},
            'real_vs_llama': {'mean': df.loc['Real', 'Llama'], 'std': 0.0, 'min': 0, 'p25': 0, 'median': 0, 'p75': 0, 'max': 0, 'sigma_mean': 0, 'sigma_std': 0},
            'qwen_vs_llama': {'mean': df.loc['Qwen', 'Llama'], 'std': 0.0, 'min': 0, 'p25': 0, 'median': 0, 'p75': 0, 'max': 0, 'sigma_mean': 0, 'sigma_std': 0},
        }
    }

dimp_diagnostics = load_dimp_diagnostics()

synthetic_vs_real = create_synthetic_vs_real_table(mmd_results)
synthetic_vs_real.to_csv(output_dir / 'synthetic_vs_real_mmd.csv', index=False)

merged_df = merge_mmd_with_dimp(synthetic_vs_real, dimp_diagnostics)
merged_df.to_csv(output_dir / 'mmd_dimp_relationship.csv', index=False)

correlation_df = compute_correlation_analysis(merged_df)
if not correlation_df.empty:
    correlation_df.to_csv(output_dir / 'mmd_correlation_results.csv', index=False)

create_figures(merged_df, mmd_dfs, output_dir)

excel_path = output_dir / 'distribution_analysis.xlsx'
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

text_path = output_dir / 'distribution_analysis.txt'
generate_analysis_text(
    word_overlap_results,
    mmd_results,
    merged_df,
    correlation_df,
    text_path
)

print('✓ Excel and figures generated')
EOF

echo ""
echo "============================================================"
echo "Distribution Analysis Complete!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - outputs/distribution_analysis/distribution_analysis.xlsx"
echo "  - outputs/distribution_analysis/distribution_analysis.txt"
echo "  - outputs/distribution_analysis/figures/*.png"
echo "  - outputs/distribution_analysis/*.csv (all data tables)"
echo ""
echo "Key results:"
cat outputs/distribution_analysis/distribution_analysis.txt
