"""Simplified distribution analysis - run step by step to avoid timeout issues."""
import argparse
from pathlib import Path

print("="*60)
print("Simplified Distribution Analysis")
print("="*60)

step = input("\nWhich step to run?\n1. Word overlap only\n2. Compute embeddings only\n3. Compute MMD only\n4. Load DIMP diagnostics\n5. Generate full report\nEnter step number: ")

if step == "1":
    print("\n--- Step 1: Word Overlap Analysis ---")
    from load_phase1_data import load_all_data
    from word_overlap import analyze_word_overlap

    output_dir = Path("outputs/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_all_data()
    word_overlap_results = analyze_word_overlap(all_data)

    word_overlap_results['df_raw'].to_csv(output_dir / "word_overlap_raw.csv")
    word_overlap_results['df_content'].to_csv(output_dir / "word_overlap_content.csv")

    print(f"\nSaved word overlap results to {output_dir}/")

elif step == "2":
    print("\n--- Step 2: Compute Embeddings ---")
    from load_phase1_data import load_all_data
    from compute_embeddings import compute_and_cache_all_embeddings

    output_dir = Path("outputs/distribution_analysis")

    all_data = load_all_data()
    embeddings_data = compute_and_cache_all_embeddings(
        all_data,
        cache_dir=output_dir / "embeddings",
        force_recompute=False
    )

    print(f"\nSaved embeddings to {output_dir}/embeddings/")

elif step == "3":
    print("\n--- Step 3: Compute MMD ---")
    from load_phase1_data import load_all_data
    from compute_embeddings import compute_and_cache_all_embeddings
    from compute_mmd import compute_all_mmd_matrices, create_mmd_dataframes

    output_dir = Path("outputs/distribution_analysis")

    # Load data and embeddings
    all_data = load_all_data()
    embeddings_data = compute_and_cache_all_embeddings(
        all_data,
        cache_dir=output_dir / "embeddings",
        force_recompute=False
    )

    # Compute MMD
    mmd_results = compute_all_mmd_matrices(
        embeddings_data,
        sample_size_overall=160,  # min(real, qwen, llama)
        sample_size_class=40,     # min per class
        num_repeats=100,
        seed=42
    )

    mmd_dfs = create_mmd_dataframes(mmd_results)

    # Save MMD CSVs
    for scope, df in mmd_dfs.items():
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"mmd_{scope}.csv")

    print(f"\nSaved MMD matrices to {output_dir}/")

elif step == "4":
    print("\n--- Step 4: Load DIMP Diagnostics ---")
    from load_dimp_diagnostics import load_dimp_diagnostics

    output_dir = Path("outputs/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = load_dimp_diagnostics()

    # Save to CSV
    diagnostics['class_level_metrics'].to_csv(output_dir / "dimp_class_metrics.csv", index=False)
    diagnostics['overall_metrics'].to_csv(output_dir / "dimp_overall_metrics.csv", index=False)

    print(f"\nSaved DIMP diagnostics to {output_dir}/")

elif step == "5":
    print("\n--- Step 5: Generate Full Report ---")
    print("This will combine all previous results into Excel and figures.")
    print("Make sure steps 1-4 have been completed first!")

    # Import all analysis functions
    from analyze_distribution import (
        create_synthetic_vs_real_table,
        merge_mmd_with_dimp,
        compute_correlation_analysis,
        create_figures,
        generate_analysis_text,
        generate_excel_report
    )
    from compute_mmd import create_mmd_dataframes
    import json
    import pandas as pd
    from pathlib import Path

    output_dir = Path("outputs/distribution_analysis")

    # Load pre-computed results
    print("\nLoading pre-computed results...")

    # Word overlap
    word_overlap_results = {
        'df_raw': pd.read_csv(output_dir / "word_overlap_raw.csv", index_col=0),
        'df_content': pd.read_csv(output_dir / "word_overlap_content.csv", index_col=0)
    }

    # MMD matrices
    mmd_dfs = {}
    for scope in ['overall', 'world', 'sports', 'business', 'sci_tech']:
        csv_path = output_dir / f"mmd_{scope}.csv"
        if csv_path.exists():
            mmd_dfs[scope] = pd.read_csv(csv_path, index_col=0)

    # Create mmd_results structure from CSVs (simplified)
    mmd_results = {}
    for scope, df in mmd_dfs.items():
        mmd_results[scope] = {
            'matrix': df.values,
            'details': {
                'real_vs_qwen': {'mean': df.loc['Real', 'Qwen'], 'std': 0.0},
                'real_vs_llama': {'mean': df.loc['Real', 'Llama'], 'std': 0.0},
                'qwen_vs_llama': {'mean': df.loc['Qwen', 'Llama'], 'std': 0.0},
            }
        }

    # DIMP diagnostics
    from load_dimp_diagnostics import load_dimp_diagnostics
    dimp_diagnostics = load_dimp_diagnostics()

    # Create tables
    print("\nCreating analysis tables...")
    synthetic_vs_real = create_synthetic_vs_real_table(mmd_results)
    synthetic_vs_real.to_csv(output_dir / "synthetic_vs_real_mmd.csv", index=False)

    merged_df = merge_mmd_with_dimp(synthetic_vs_real, dimp_diagnostics)
    merged_df.to_csv(output_dir / "mmd_dimp_relationship.csv", index=False)

    correlation_df = compute_correlation_analysis(merged_df)
    if not correlation_df.empty:
        correlation_df.to_csv(output_dir / "mmd_correlation_results.csv", index=False)

    # Generate figures
    print("\nGenerating figures...")
    create_figures(merged_df, mmd_dfs, output_dir)

    # Generate Excel
    print("\nGenerating Excel report...")
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

    # Generate analysis text
    print("\nGenerating analysis text...")
    text_path = output_dir / "distribution_analysis.txt"
    generate_analysis_text(
        word_overlap_results,
        mmd_results,
        merged_df,
        correlation_df,
        text_path
    )

    print("\n" + "="*60)
    print("Full Report Generation Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}/")

else:
    print("Invalid step number!")
