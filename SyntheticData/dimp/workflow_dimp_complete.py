"""Complete workflow for DIMP training with QWEN and Llama.

This script provides a complete workflow from data generation to DIMP training.

Workflow:
1. Prepare real AG News data (train/test splits)
2. Generate QWEN synthetic data
3. Generate Llama synthetic data
4. Verify all data formats
5. Train DIMP models and evaluate

Usage:
  # Full workflow (you need to provide model paths)
  python workflow_dimp_complete.py \
    --qwen-model-dir /path/to/qwen \
    --llama-model-dir /path/to/llama \
    --output-dir outputs/dimp_experiment \
    --num-per-label 100

  # Skip data generation if you already have it
  python workflow_dimp_complete.py \
    --skip-generation \
    --qwen-data outputs/protocol/synthetic_qwen.jsonl \
    --llama-data outputs/protocol/synthetic_llama.jsonl \
    --real-train outputs/protocol/real_train.jsonl \
    --real-test outputs/protocol/real_test.jsonl \
    --output-dir outputs/dimp_experiment
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with code {result.returncode}")
        return result.returncode

    print(f"\n✓ {description} completed successfully")
    return 0


def prepare_real_data(output_dir: Path, train_per_label: int, test_per_label: int, seed: int) -> dict[str, str]:
    """Prepare real AG News data splits."""
    print("\n" + "="*60)
    print("Step 1: Preparing Real AG News Data")
    print("="*60)

    protocol_dir = output_dir / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    real_train = protocol_dir / "real_train.jsonl"
    real_test = protocol_dir / "real_test.jsonl"

    # Check if already exists
    if real_train.exists() and real_test.exists():
        print(f"✓ Real data already exists:")
        print(f"  - {real_train}")
        print(f"  - {real_test}")
        return {
            "real_train": str(real_train),
            "real_test": str(real_test),
        }

    # Use existing synthetic_agnews.py prepare command
    cmd = [
        sys.executable,
        "synthetic_agnews.py",
        "prepare",
        "--protocol-dir", str(protocol_dir),
        "--real-train-per-label", str(train_per_label),
        "--test-per-label", str(test_per_label),
        "--seed", str(seed),
    ]

    result = run_command(cmd, "Preparing real AG News data")
    if result != 0:
        raise RuntimeError("Failed to prepare real data")

    return {
        "real_train": str(real_train),
        "real_test": str(real_test),
    }


def generate_qwen_data(
    qwen_model_dir: str,
    output_path: Path,
    real_train_path: str,
    num_per_label: int,
    seed: int,
    use_reference: bool,
) -> str:
    """Generate QWEN synthetic data."""
    print("\n" + "="*60)
    print("Step 2: Generating QWEN Synthetic Data")
    print("="*60)

    if output_path.exists():
        print(f"✓ QWEN data already exists: {output_path}")
        return str(output_path)

    # Use existing synthetic_agnews.py generate command
    cmd = [
        sys.executable,
        "synthetic_agnews.py",
        "generate",
        "--model", "qwen",
        "--qwen-dir", qwen_model_dir,
        "--output", str(output_path),
        "--num-per-label", str(num_per_label),
        "--seed", str(seed),
    ]

    if use_reference:
        cmd.extend([
            "--use-reference",
            "--protocol-dir", str(output_path.parent),
        ])

    result = run_command(cmd, "Generating QWEN synthetic data")
    if result != 0:
        raise RuntimeError("Failed to generate QWEN data")

    return str(output_path)


def generate_llama_data(
    llama_model_dir: str,
    output_path: Path,
    real_train_path: str,
    num_per_label: int,
    seed: int,
    use_reference: bool,
    use_4bit: bool,
) -> str:
    """Generate Llama synthetic data."""
    print("\n" + "="*60)
    print("Step 3: Generating Llama Synthetic Data")
    print("="*60)

    if output_path.exists():
        print(f"✓ Llama data already exists: {output_path}")
        return str(output_path)

    cmd = [
        sys.executable,
        "generate_llama_synthetic.py",
        "--llama-dir", llama_model_dir,
        "--output", str(output_path),
        "--num-per-label", str(num_per_label),
        "--seed", str(seed),
    ]

    if use_reference:
        cmd.extend([
            "--real-train", real_train_path,
            "--use-reference",
        ])

    if use_4bit:
        cmd.append("--use-4bit")

    result = run_command(cmd, "Generating Llama synthetic data")
    if result != 0:
        raise RuntimeError("Failed to generate Llama data")

    return str(output_path)


def verify_data(
    qwen_data: str,
    llama_data: str,
    real_train: str,
    real_test: str,
) -> bool:
    """Verify all data formats."""
    print("\n" + "="*60)
    print("Step 4: Verifying Data Formats")
    print("="*60)

    cmd = [
        sys.executable,
        "verify_data_for_dimp.py",
        "--qwen-data", qwen_data,
        "--llama-data", llama_data,
        "--real-train", real_train,
        "--real-test", real_test,
    ]

    result = run_command(cmd, "Verifying data formats")
    return result == 0


def update_training_script(
    qwen_data: str,
    llama_data: str,
    real_train: str,
    real_test: str,
) -> None:
    """Update placeholder paths in training script."""
    print("\n" + "="*60)
    print("Step 5: Updating Training Script Paths")
    print("="*60)

    script_path = Path("train_dimp_llama_qwen.py")

    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Read current content
    content = script_path.read_text(encoding="utf-8")

    # Replace placeholders
    replacements = {
        'QWEN_SYNTHETIC_PATH = "PLACEHOLDER_QWEN_SYNTHETIC.jsonl"': f'QWEN_SYNTHETIC_PATH = "{qwen_data}"',
        'LLAMA_SYNTHETIC_PATH = "PLACEHOLDER_LLAMA_SYNTHETIC.jsonl"': f'LLAMA_SYNTHETIC_PATH = "{llama_data}"',
        'REAL_TRAIN_PATH = "PLACEHOLDER_REAL_TRAIN.jsonl"': f'REAL_TRAIN_PATH = "{real_train}"',
        'REAL_TEST_PATH = "PLACEHOLDER_REAL_TEST.jsonl"': f'REAL_TEST_PATH = "{real_test}"',
    }

    updated = content
    for old, new in replacements.items():
        if old in updated:
            updated = updated.replace(old, new)
            print(f"✓ Updated: {old.split('=')[0].strip()}")

    # Check if any placeholders remain
    if "PLACEHOLDER" in updated:
        print("⚠ WARNING: Some placeholders still remain in the script")

    # Save updated script
    script_path.write_text(updated, encoding="utf-8")
    print(f"✓ Training script updated: {script_path}")


def run_dimp_training(output_dir: Path, args: argparse.Namespace) -> None:
    """Run DIMP training experiments."""
    print("\n" + "="*60)
    print("Step 6: Running DIMP Training")
    print("="*60)

    cmd = [
        sys.executable,
        "train_dimp_llama_qwen.py",
        "--protocol-dir", str(output_dir / "dimp_results"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--gamma", str(args.gamma),
        "--seed", str(args.seed),
    ]

    if args.reuse_qc:
        cmd.append("--reuse-qc")

    result = run_command(cmd, "Running DIMP training experiments")
    if result != 0:
        raise RuntimeError("DIMP training failed")


def print_summary(output_dir: Path) -> None:
    """Print summary of results."""
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*60)

    results_path = output_dir / "dimp_results" / "all_results.json"

    if results_path.exists():
        results = json.loads(results_path.read_text())

        print("\nExperiment Results:")
        print(f"{'Experiment':<35} {'Accuracy':<10} {'Macro F1':<10}")
        print("-" * 60)

        for r in results:
            acc = r['metrics']['eval_accuracy']
            f1 = r['metrics']['eval_macro_f1']
            print(f"{r['run']:<35} {acc:<10.4f} {f1:<10.4f}")

        print(f"\nFull results saved to: {results_path}")
    else:
        print(f"\n⚠ Results file not found: {results_path}")

    print(f"\nAll outputs saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete DIMP training workflow with QWEN and Llama"
    )

    # Data generation options
    gen_group = parser.add_argument_group("Data Generation")
    gen_group.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip data generation (use existing data)",
    )
    gen_group.add_argument(
        "--qwen-model-dir",
        type=str,
        help="Path to QWEN model (required if not skipping generation)",
    )
    gen_group.add_argument(
        "--llama-model-dir",
        type=str,
        help="Path to Llama model (required if not skipping generation)",
    )
    gen_group.add_argument(
        "--num-per-label",
        type=int,
        default=100,
        help="Synthetic samples per label (default: 100)",
    )
    gen_group.add_argument(
        "--use-reference",
        action="store_true",
        help="Use real examples as style references during generation",
    )
    gen_group.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization for Llama",
    )

    # Existing data paths (if skipping generation)
    data_group = parser.add_argument_group("Existing Data Paths")
    data_group.add_argument(
        "--qwen-data",
        type=str,
        help="Path to existing QWEN data (if --skip-generation)",
    )
    data_group.add_argument(
        "--llama-data",
        type=str,
        help="Path to existing Llama data (if --skip-generation)",
    )
    data_group.add_argument(
        "--real-train",
        type=str,
        help="Path to existing real train data (if --skip-generation)",
    )
    data_group.add_argument(
        "--real-test",
        type=str,
        help="Path to existing real test data (if --skip-generation)",
    )

    # Training options
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    train_group.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="DIMP gamma parameter (default: 1.0)",
    )
    train_group.add_argument(
        "--reuse-qc",
        action="store_true",
        help="Reuse existing Quality Checker",
    )

    # General options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dimp_experiment",
        help="Output directory (default: outputs/dimp_experiment)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (only generate/verify data)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.skip_generation:
        if not args.qwen_model_dir or not args.llama_model_dir:
            parser.error("--qwen-model-dir and --llama-model-dir are required unless --skip-generation is set")
    else:
        if not all([args.qwen_data, args.llama_data, args.real_train, args.real_test]):
            parser.error("All data paths (--qwen-data, --llama-data, --real-train, --real-test) "
                        "are required when --skip-generation is set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1-3: Generate or use existing data
        if args.skip_generation:
            print("Skipping data generation, using existing data...")
            data_paths = {
                "qwen_data": args.qwen_data,
                "llama_data": args.llama_data,
                "real_train": args.real_train,
                "real_test": args.real_test,
            }
        else:
            # Step 1: Prepare real data
            real_data = prepare_real_data(
                output_dir,
                train_per_label=100,
                test_per_label=20,
                seed=args.seed,
            )

            # Step 2: Generate QWEN data
            qwen_data = generate_qwen_data(
                args.qwen_model_dir,
                output_dir / "protocol" / "synthetic_qwen.jsonl",
                real_data["real_train"],
                args.num_per_label,
                args.seed,
                args.use_reference,
            )

            # Step 3: Generate Llama data
            llama_data = generate_llama_data(
                args.llama_model_dir,
                output_dir / "protocol" / "synthetic_llama.jsonl",
                real_data["real_train"],
                args.num_per_label,
                args.seed + 1000,  # Different seed for diversity
                args.use_reference,
                args.use_4bit,
            )

            data_paths = {
                "qwen_data": qwen_data,
                "llama_data": llama_data,
                "real_train": real_data["real_train"],
                "real_test": real_data["real_test"],
            }

        # Step 4: Verify data
        data_ok = verify_data(
            data_paths["qwen_data"],
            data_paths["llama_data"],
            data_paths["real_train"],
            data_paths["real_test"],
        )

        if not data_ok:
            print("\n❌ Data verification failed. Please fix data issues before training.")
            return 1

        # Step 5: Update training script
        update_training_script(
            data_paths["qwen_data"],
            data_paths["llama_data"],
            data_paths["real_train"],
            data_paths["real_test"],
        )

        if args.skip_training:
            print("\nSkipping training as requested.")
            print("Data is ready. Run train_dimp_llama_qwen.py when ready to train.")
            return 0

        # Step 6: Run DIMP training
        run_dimp_training(output_dir, args)

        # Print summary
        print_summary(output_dir)

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
