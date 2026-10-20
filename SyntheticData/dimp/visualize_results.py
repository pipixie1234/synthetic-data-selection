"""Visualize DIMP experiment results with Excel tables and loss curves.

This script:
1. Reads all experiment results from the protocol directory
2. Creates an Excel summary table with all metrics
3. Plots loss curves during training (if available)
4. Generates comparison charts for accuracy and F1 scores

Usage:
    python visualize_results.py --protocol-dir outputs/protocol_dimp_llama_qwen
"""
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_results(protocol_dir: Path) -> list[dict[str, Any]]:
    """Load all experiment results."""
    results_path = protocol_dir / "all_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open() as f:
        return json.load(f)


def load_training_logs(protocol_dir: Path, run_name: str) -> dict[str, list]:
    """Load training logs if available (loss per epoch)."""
    log_path = protocol_dir / run_name / "trainer_state.json"
    if not log_path.exists():
        return {}

    with log_path.open() as f:
        state = json.load(f)

    # Extract loss history
    logs = {"epoch": [], "train_loss": [], "eval_loss": [], "eval_accuracy": []}

    for entry in state.get("log_history", []):
        if "epoch" in entry:
            epoch = entry["epoch"]
            if "loss" in entry:
                logs["epoch"].append(epoch)
                logs["train_loss"].append(entry["loss"])
            if "eval_loss" in entry:
                if epoch not in logs["epoch"]:
                    logs["epoch"].append(epoch)
                logs["eval_loss"].append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                logs["eval_accuracy"].append(entry["eval_accuracy"])

    return logs


def create_excel_summary(results: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Create Excel summary table with all experiment results."""
    rows = []

    for result in results:
        run_name = result["run"]
        metrics = result["metrics"]
        diagnostics = result.get("diagnostics", {})

        row = {
            "Experiment": run_name,
            "Loss Type": result.get("loss_type", "N/A"),
            "Gamma": result.get("gamma", "N/A"),
            "Train Count": result["train_count"],
            "Test Count": result["test_count"],
            "Accuracy": metrics.get("eval_accuracy", 0),
            "Macro F1": metrics.get("eval_macro_f1", 0),
            "Loss": metrics.get("eval_loss", 0),
        }

        # Add diagnostics if available
        if diagnostics:
            row.update({
                "Avg Weight": diagnostics.get("avg_weight", "N/A"),
                "Min Weight": diagnostics.get("min_weight", "N/A"),
                "Max Weight": diagnostics.get("max_weight", "N/A"),
                "Weighted Loss": diagnostics.get("weighted_loss", "N/A"),
            })

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by experiment name
    df = df.sort_values("Experiment")

    # Save to Excel with formatting
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


def plot_accuracy_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy comparison across experiments."""
    fig, ax = plt.subplots(figsize=(14, 6))

    experiments = df["Experiment"].tolist()
    accuracies = df["Accuracy"].tolist()
    f1_scores = df["Macro F1"].tolist()

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8, color='darkorange')

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('DIMP Experiments: Accuracy and F1 Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison plot saved to: {output_path}")


def plot_loss_curves(protocol_dir: Path, results: list[dict[str, Any]], output_pdf: Path) -> None:
    """Plot training and evaluation loss curves for all experiments."""
    with PdfPages(output_pdf) as pdf:
        for result in results:
            run_name = result["run"]
            logs = load_training_logs(protocol_dir, run_name)

            if not logs or not logs["epoch"]:
                print(f"No training logs found for {run_name}, skipping...")
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Loss curves
            ax1 = axes[0]
            if logs["train_loss"]:
                ax1.plot(logs["epoch"][:len(logs["train_loss"])], logs["train_loss"],
                        marker='o', label='Train Loss', linewidth=2, markersize=6)
            if logs["eval_loss"]:
                eval_epochs = logs["epoch"][:len(logs["eval_loss"])]
                ax1.plot(eval_epochs, logs["eval_loss"],
                        marker='s', label='Eval Loss', linewidth=2, markersize=6)

            ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax1.set_title(f'{run_name}: Loss Curves', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)

            # Plot 2: Accuracy curve
            ax2 = axes[1]
            if logs["eval_accuracy"]:
                eval_epochs = logs["epoch"][:len(logs["eval_accuracy"])]
                ax2.plot(eval_epochs, logs["eval_accuracy"],
                        marker='D', label='Eval Accuracy', linewidth=2,
                        markersize=6, color='green')

            ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax2.set_title(f'{run_name}: Accuracy Curve', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"Loss curves saved to: {output_pdf}")


def plot_weight_distribution(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot DIMP weight statistics across experiments."""
    experiments = []
    avg_weights = []
    min_weights = []
    max_weights = []

    for result in results:
        diagnostics = result.get("diagnostics", {})
        if diagnostics and "avg_weight" in diagnostics:
            experiments.append(result["run"])
            avg_weights.append(diagnostics["avg_weight"])
            min_weights.append(diagnostics["min_weight"])
            max_weights.append(diagnostics["max_weight"])

    if not experiments:
        print("No weight diagnostics available, skipping weight distribution plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(experiments))

    ax.plot(x, avg_weights, marker='o', label='Avg Weight', linewidth=2, markersize=8)
    ax.fill_between(x, min_weights, max_weights, alpha=0.2, label='Min-Max Range')

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('DIMP Weight', fontsize=12, fontweight='bold')
    ax.set_title('DIMP Weight Distribution Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Weight distribution plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DIMP experiment results")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default="outputs/protocol_dimp_llama_qwen",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (defaults to protocol-dir/visualizations)",
    )

    args = parser.parse_args()

    protocol_dir = Path(args.protocol_dir)
    output_dir = Path(args.output_dir) if args.output_dir else protocol_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {protocol_dir}")
    results = load_results(protocol_dir)
    print(f"Found {len(results)} experiments")

    # Create Excel summary
    print("\n=== Creating Excel Summary ===")
    df = create_excel_summary(results, output_dir / "results_summary.xlsx")
    print("\nSummary Table:")
    print(df.to_string(index=False))

    # Plot accuracy comparison
    print("\n=== Creating Accuracy Comparison Plot ===")
    plot_accuracy_comparison(df, output_dir / "accuracy_comparison.png")

    # Plot loss curves
    print("\n=== Creating Loss Curves ===")
    plot_loss_curves(protocol_dir, results, output_dir / "loss_curves.pdf")

    # Plot weight distribution
    print("\n=== Creating Weight Distribution Plot ===")
    plot_weight_distribution(results, output_dir / "weight_distribution.png")

    print(f"\n{'='*60}")
    print("All visualizations completed!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - results_summary.xlsx")
    print(f"  - accuracy_comparison.png")
    print(f"  - loss_curves.pdf")
    print(f"  - weight_distribution.png")


if __name__ == "__main__":
    main()
