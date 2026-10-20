"""Visualize SmaExp comparison experiment results.

This script:
1. Reads comparison results from the protocol directory
2. Creates Excel summary with all 8 experiments
3. Plots accuracy and F1 comparisons
4. Generates grouped bar charts for different experiment types

Usage:
    python visualize_comparison_results.py --results outputs/comparison_results.json
"""
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load comparison results."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open() as f:
        return json.load(f)


def create_excel_summary(results: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Create Excel summary table."""
    rows = []

    for result in results:
        run_name = result["run"]
        metrics = result["metrics"]

        row = {
            "Experiment": run_name,
            "Train Count": result["train_count"],
            "Test Count": result["test_count"],
            "Accuracy": metrics.get("eval_accuracy", 0),
            "Macro F1": metrics.get("eval_macro_f1", 0),
            "Loss": metrics.get("eval_loss", 0),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by experiment name
    df = df.sort_values("Experiment")

    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['Results']
        for i, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + i)].width = min(max_length + 2, 50)

    print(f"Excel summary saved to: {output_path}")
    return df


def plot_grouped_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Plot grouped comparison: Exp1-4 vs Exp5-8."""
    # Split experiments into two groups
    exp1_4 = df[df["Experiment"].str.contains("exp[1-4]")]
    exp5_8 = df[df["Experiment"].str.contains("exp[5-8]")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Exp1-4 (test on real data)
    ax1 = axes[0]
    experiments = exp1_4["Experiment"].tolist()
    accuracies = exp1_4["Accuracy"].tolist()
    f1_scores = exp1_4["Macro F1"].tolist()

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8, color='darkorange')

    ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Experiments 1-4: Test on Real Data', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.replace("_", "\n") for e in experiments], fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    # Plot Exp5-8 (synthetic cross-validation)
    ax2 = axes[1]
    experiments = exp5_8["Experiment"].tolist()
    accuracies = exp5_8["Accuracy"].tolist()
    f1_scores = exp5_8["Macro F1"].tolist()

    x = np.arange(len(experiments))

    bars1 = ax2.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
    bars2 = ax2.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8, color='darkorange')

    ax2.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Experiments 5-8: Synthetic Cross-Validation', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.replace("_", "\n") for e in experiments], fontsize=9)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grouped comparison plot saved to: {output_path}")


def plot_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot heatmap of accuracy across experiments."""
    # Extract experiment info
    exp_names = df["Experiment"].tolist()
    accuracies = df["Accuracy"].tolist()

    # Create matrix for heatmap
    # Group 1: Exp1-4, Group 2: Exp5-8
    matrix_data = []
    matrix_labels = []

    for i, (name, acc) in enumerate(zip(exp_names, accuracies)):
        if "exp1" in name or "exp2" in name or "exp3" in name or "exp4" in name:
            group = "Test on Real"
        else:
            group = "Synthetic Test"
        matrix_data.append([acc])
        matrix_labels.append(name)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Reshape for better visualization
    data_array = np.array(matrix_data)

    im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_yticks(np.arange(len(matrix_labels)))
    ax.set_yticklabels(matrix_labels)
    ax.set_xticks([0])
    ax.set_xticklabels(['Accuracy'])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(matrix_labels)):
        text = ax.text(0, i, f'{data_array[i, 0]:.3f}',
                      ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Accuracy Heatmap: All Experiments', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def plot_model_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Compare performance across different training sources."""
    # Extract model-specific results
    real_row = df[df["Experiment"].str.contains("real50")]
    qwen_row = df[df["Experiment"].str.contains("qwen50")]
    llama_row = df[df["Experiment"].str.contains("llama50")]
    mixed_row = df[df["Experiment"].str.contains("mixed50")]

    models = ["Real", "Qwen", "Llama", "Mixed"]
    accuracies = [
        real_row["Accuracy"].values[0] if len(real_row) > 0 else 0,
        qwen_row["Accuracy"].values[0] if len(qwen_row) > 0 else 0,
        llama_row["Accuracy"].values[0] if len(llama_row) > 0 else 0,
        mixed_row["Accuracy"].values[0] if len(mixed_row) > 0 else 0,
    ]
    f1_scores = [
        real_row["Macro F1"].values[0] if len(real_row) > 0 else 0,
        qwen_row["Macro F1"].values[0] if len(qwen_row) > 0 else 0,
        llama_row["Macro F1"].values[0] if len(llama_row) > 0 else 0,
        mixed_row["Macro F1"].values[0] if len(mixed_row) > 0 else 0,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8, color='darkorange')

    ax.set_xlabel('Training Data Source', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by Training Source (50 samples/class)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SmaExp comparison results")
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/comparison_results.json",
        help="Path to comparison results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (defaults to same dir as results)",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    print(f"Found {len(results)} experiments")

    # Create Excel summary
    print("\n=== Creating Excel Summary ===")
    df = create_excel_summary(results, output_dir / "comparison_summary.xlsx")
    print("\nSummary Table:")
    print(df.to_string(index=False))

    # Plot grouped comparison
    print("\n=== Creating Grouped Comparison Plot ===")
    plot_grouped_comparison(df, output_dir / "grouped_comparison.png")

    # Plot heatmap
    print("\n=== Creating Heatmap ===")
    plot_heatmap(df, output_dir / "accuracy_heatmap.png")

    # Plot model comparison
    print("\n=== Creating Model Comparison Plot ===")
    plot_model_comparison(df, output_dir / "model_comparison.png")

    print(f"\n{'='*60}")
    print("All visualizations completed!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - comparison_summary.xlsx")
    print(f"  - grouped_comparison.png")
    print(f"  - accuracy_heatmap.png")
    print(f"  - model_comparison.png")


if __name__ == "__main__":
    main()
