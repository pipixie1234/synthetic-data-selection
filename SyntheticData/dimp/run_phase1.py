"""Phase 1 complete experiment runner.

This script runs all Phase 1 experiments with proper data splits, QC consistency checks,
and comprehensive diagnostics collection.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_splits import (
    apply_splits,
    create_fixed_splits,
    load_splits,
    save_splits,
    verify_splits_no_leakage,
)
from phase1_trainers import Phase1Trainer, train_with_epoch_callbacks
from enhanced_diagnostics import (
    compute_full_sample_diagnostics,
    aggregate_epoch_diagnostics,
)
from qc_consistency_check import check_qc_consistency, save_consistency_check_results
from train_dimp_llama_qwen import QualityChecker, load_jsonl

# AG News labels
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Data paths
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = ROOT / "SyntheticData" / "SmaExp" / "outputs" / "protocol"
QWEN_SYNTHETIC_PATH = PROTOCOL_DIR / "synthetic_qwen.jsonl"
LLAMA_SYNTHETIC_PATH = PROTOCOL_DIR / "synthetic_llama.jsonl"
REAL_TRAIN_PATH = PROTOCOL_DIR / "real_train.jsonl"
REAL_TEST_PATH = PROTOCOL_DIR / "real_test.jsonl"


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_create_splits(
    qwen_full, llama_full, real_train_full, real_test_full,
    splits_dir: Path, seed: int, force_recreate: bool = False
) -> tuple[dict, dict]:
    """Load existing splits or create new ones.

    Returns:
        (splits, datasets)
    """
    if splits_dir.exists() and not force_recreate:
        print(f"\nLoading existing splits from: {splits_dir}")
        splits = load_splits(splits_dir)

        # Verify no leakage
        print("\nVerifying splits for data leakage...")
        leakage_checks = verify_splits_no_leakage(splits)
        for check_name, passed in leakage_checks.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {check_name}: {status}")

        if not leakage_checks['all_passed']:
            raise RuntimeError("Data leakage detected in splits!")

    else:
        print(f"\nCreating new splits with seed={seed}")
        splits = create_fixed_splits(
            qwen_full, llama_full, real_train_full, real_test_full,
            seed=seed,
            qc_train_per_class=40,
            qc_eval_per_class=10,
            synthetic_train_per_class=80,
            synthetic_test_per_class=20,
        )

        splits_dir.mkdir(parents=True, exist_ok=True)
        save_splits(splits, splits_dir)

        print("\nVerifying splits for data leakage...")
        leakage_checks = verify_splits_no_leakage(splits)
        for check_name, passed in leakage_checks.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {check_name}: {status}")

    # Apply splits to get datasets
    datasets = apply_splits(qwen_full, llama_full, real_train_full, real_test_full, splits)

    print("\nDataset sizes:")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)}")

    return splits, datasets


def train_or_load_qc(datasets, protocol_dir, args, force_retrain=False):
    """Train Quality Checker or load existing one."""
    qc_dir = protocol_dir / "quality_checker"

    if qc_dir.exists() and not force_retrain:
        print(f"\nReusing existing Quality Checker from: {qc_dir}")
        qc = QualityChecker(model_name=args.bert_model, num_labels=4)
        qc.load(qc_dir)
    else:
        print(f"\nTraining Quality Checker...")
        qc = QualityChecker(model_name=args.bert_model, num_labels=4)
        qc.train(
            datasets['real_qc_train'],
            datasets['real_qc_eval'],
            protocol_dir,
            epochs=args.qc_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    return qc, qc_dir


def run_qc_consistency_check(qc, datasets, qc_dir, protocol_dir):
    """Run QC save/load consistency check."""
    print(f"\n{'='*60}")
    print("Running QC Consistency Check")
    print(f"{'='*60}")

    consistency_results = check_qc_consistency(
        qc,
        datasets['real_qc_eval'],
        qc_dir,
        batch_size=32,
        max_length=256,
    )

    # Save results
    check_path = protocol_dir / "qc_reload_check.json"
    save_consistency_check_results(consistency_results, check_path)

    if not consistency_results['passed']:
        print("\n" + "!"*60)
        print("WARNING: QC consistency check FAILED!")
        print("Please investigate before running experiments.")
        print("!"*60)

        if not input("\nContinue anyway? (yes/no): ").lower().startswith('y'):
            raise RuntimeError("QC consistency check failed. Aborting.")

    return consistency_results


def test_qc_only(qc, datasets, protocol_dir, args):
    """Test Quality Checker directly on Real Test."""
    print(f"\n{'='*60}")
    print("Experiment: QC-only (160 real train) → Real Test")
    print(f"{'='*60}")

    from datasets import Dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import AutoTokenizer
    import torch

    real_test = datasets['real_test']
    test_ds = Dataset.from_list([
        {"text": row["text"], "labels": int(row["label"])}
        for row in real_test
    ])

    tokenizer = AutoTokenizer.from_pretrained(qc.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    # Get predictions
    qc.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qc.model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(test_ds), 32):
            batch = test_ds[i:i+32]
            inputs = tokenizer(
                [real_test[j]['text'] for j in range(i, min(i+32, len(real_test)))],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors='pt'
            ).to(device)

            outputs = qc.model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch['labels']

            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Handle cases where not all classes appear
    unique_labels = sorted(set(all_labels) | set(all_preds))
    target_names_filtered = [LABELS[i] for i in unique_labels]

    report = classification_report(
        all_labels, all_preds, labels=unique_labels, target_names=target_names_filtered,
        digits=4, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        all_labels, all_preds, labels=unique_labels, target_names=target_names_filtered,
        digits=4, zero_division=0
    )

    results = {
        "run": "QC-only_160real",
        "loss_type": "ce",
        "train_count": 160,
        "test_count": len(real_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_metrics": report,
    }

    # Save results
    qc_only_dir = protocol_dir / "QC-only_160real"
    qc_only_dir.mkdir(parents=True, exist_ok=True)
    (qc_only_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (qc_only_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    print(f"\nQC-only results: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}")
    print(report_text)

    return results


def compute_and_cache_quality_scores(qc, datasets, protocol_dir, force_recompute=False):
    """Compute quality scores for synthetic training data."""
    cache_dir = protocol_dir / "quality_scores"
    cache_dir.mkdir(parents=True, exist_ok=True)

    qwen_cache = cache_dir / "qwen_train_quality_scores.json"
    llama_cache = cache_dir / "llama_train_quality_scores.json"

    # Qwen quality scores
    if qwen_cache.exists() and not force_recompute:
        print(f"\nLoading cached Qwen quality scores from: {qwen_cache}")
        with qwen_cache.open() as f:
            qwen_quality_scores = json.load(f)
    else:
        print(f"\nComputing quality scores for Qwen training data...")
        qwen_quality_scores = qc.compute_quality_scores(
            datasets['qwen_train'], batch_size=32, max_length=256
        )
        with qwen_cache.open('w') as f:
            json.dump(qwen_quality_scores, f, indent=2)

    # Llama quality scores
    if llama_cache.exists() and not force_recompute:
        print(f"\nLoading cached Llama quality scores from: {llama_cache}")
        with llama_cache.open() as f:
            llama_quality_scores = json.load(f)
    else:
        print(f"\nComputing quality scores for Llama training data...")
        llama_quality_scores = qc.compute_quality_scores(
            datasets['llama_train'], batch_size=32, max_length=256
        )
        with llama_cache.open('w') as f:
            json.dump(llama_quality_scores, f, indent=2)

    return qwen_quality_scores, llama_quality_scores


def run_single_seed_experiments(
    datasets,
    qwen_quality_scores,
    llama_quality_scores,
    protocol_dir,
    seed,
    args,
):
    """Run all experiments for a single seed."""
    print(f"\n{'='*60}")
    print(f"Running experiments with seed={seed}")
    print(f"{'='*60}")

    seed_everything(seed)
    results = []

    if args.dimp_only:
        # Diagnostics rerun: preserve existing CE/Quality/Real results and
        # only train the two DIMP models with saved checkpoints.
        for source, quality_scores, prefix in [
            ("qwen", qwen_quality_scores, "B3_Qwen320_DIMP"),
            ("llama", llama_quality_scores, "C3_Llama320_DIMP"),
        ]:
            print(f"\n--- {prefix} ---")
            trainer = Phase1Trainer(
                model_name=args.bert_model,
                loss_type="dimp",
                gamma=args.gamma,
                epsilon=args.epsilon,
            )
            result = trainer.train(
                datasets[f"{source}_train"],
                datasets["real_test"],
                quality_scores,
                protocol_dir,
                f"{prefix}_seed{seed}",
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                checkpoint_dir=Path(args.checkpoint_dir),
            )
            results.append(result)
        return results

    # A1: Real200 CE
    print(f"\n--- A1: Real200 CE ---")
    trainer_real = Phase1Trainer(
        model_name=args.bert_model,
        loss_type="ce",
    )
    result_a1 = trainer_real.train(
        datasets['real_train_full'],
        datasets['real_test'],
        None,
        protocol_dir,
        f"A1_Real200_CE_seed{seed}",
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result_a1)

    # B1: Qwen320 CE
    print(f"\n--- B1: Qwen320 CE ---")
    trainer_qwen_ce = Phase1Trainer(
        model_name=args.bert_model,
        loss_type="ce",
    )
    result_b1 = trainer_qwen_ce.train(
        datasets['qwen_train'],
        datasets['real_test'],
        None,
        protocol_dir,
        f"B1_Qwen320_CE_seed{seed}",
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result_b1)

    # B2: Qwen320 Quality-only
    print(f"\n--- B2: Qwen320 Quality-only ---")
    trainer_qwen_quality = Phase1Trainer(
        model_name=args.bert_model,
        loss_type="quality",
    )
    result_b2 = trainer_qwen_quality.train(
        datasets['qwen_train'],
        datasets['real_test'],
        qwen_quality_scores,
        protocol_dir,
        f"B2_Qwen320_Quality_seed{seed}",
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result_b2)

    # B3: Qwen320 DIMP
    print(f"\n--- B3: Qwen320 DIMP ---")
    trainer_qwen_dimp = Phase1Trainer(
        model_name=args.bert_model,
        loss_type="dimp",
        gamma=args.gamma,
        epsilon=args.epsilon,
    )
    result_b3 = trainer_qwen_dimp.train(
        datasets['qwen_train'],
        datasets['real_test'],
        qwen_quality_scores,
        protocol_dir,
        f"B3_Qwen320_DIMP_seed{seed}",
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    results.append(result_b3)

    # C1: Llama320 CE
    print(f"\n--- C1: Llama320 CE ---")
    trainer_llama_ce = Phase1Trainer(
        model_name=args.bert_model,
        loss_type="ce",
    )
    result_c1 = trainer_llama_ce.train(
        datasets['llama_train'],
        datasets['real_test'],
        None,
        protocol_dir,
        f"C1_Llama320_CE_seed{seed}",
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result_c1)

    # C2: Llama320 Quality-only
    print(f"\n--- C2: Llama320 Quality-only ---")
    trainer_llama_quality = Phase1Trainer(
        model_name=args.bert_model,
        loss_type="quality",
    )
    result_c2 = trainer_llama_quality.train(
        datasets['llama_train'],
        datasets['real_test'],
        llama_quality_scores,
        protocol_dir,
        f"C2_Llama320_Quality_seed{seed}",
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result_c2)

    # C3: Llama320 DIMP
    print(f"\n--- C3: Llama320 DIMP ---")
    trainer_llama_dimp = Phase1Trainer(
        model_name=args.bert_model,
        loss_type="dimp",
        gamma=args.gamma,
        epsilon=args.epsilon,
    )
    result_c3 = trainer_llama_dimp.train(
        datasets['llama_train'],
        datasets['real_test'],
        llama_quality_scores,
        protocol_dir,
        f"C3_Llama320_DIMP_seed{seed}",
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    results.append(result_c3)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 DIMP experiments")
    parser.add_argument("--protocol-dir", type=str, default="outputs/phase1")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(ROOT / "cloud" / "model" / "extra_data" / "phase1_checkpoints"),
        help="External directory for DIMP model checkpoints only",
    )
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--qc-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--force-recreate-splits", action="store_true")
    parser.add_argument("--force-retrain-qc", action="store_true")
    parser.add_argument("--force-recompute-quality", action="store_true")
    parser.add_argument("--skip-qc-check", action="store_true")
    parser.add_argument(
        "--dimp-only",
        action="store_true",
        help="Only rerun Qwen/Llama DIMP and save checkpoints for diagnostics",
    )

    args = parser.parse_args()
    protocol_dir = Path(args.protocol_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Phase 1: DIMP Fair Comparison Experiments")
    print("="*60)

    # Load data
    qwen_full = load_jsonl(QWEN_SYNTHETIC_PATH)
    llama_full = load_jsonl(LLAMA_SYNTHETIC_PATH)
    real_train_full = load_jsonl(REAL_TRAIN_PATH)
    real_test_full = load_jsonl(REAL_TEST_PATH)

    # Load or create splits
    splits_dir = protocol_dir / "splits"
    splits, datasets = load_or_create_splits(
        qwen_full, llama_full, real_train_full, real_test_full,
        splits_dir, args.seeds[0], args.force_recreate_splits
    )

    # Train or load QC
    qc, qc_dir = train_or_load_qc(datasets, protocol_dir, args, args.force_retrain_qc)

    # QC consistency check
    if not args.skip_qc_check:
        run_qc_consistency_check(qc, datasets, qc_dir, protocol_dir)

    # Test QC-only only during the complete Phase 1 workflow.
    qc_only_results = None if args.dimp_only else test_qc_only(qc, datasets, protocol_dir, args)

    # Compute quality scores
    qwen_quality_scores, llama_quality_scores = compute_and_cache_quality_scores(
        qc, datasets, protocol_dir, args.force_recompute_quality
    )

    # Run experiments for each seed
    all_results = [] if args.dimp_only else [qc_only_results]
    for seed in args.seeds:
        seed_results = run_single_seed_experiments(
            datasets, qwen_quality_scores, llama_quality_scores,
            protocol_dir, seed, args
        )
        all_results.extend(seed_results)

    # Save results
    results_path = protocol_dir / (
        "dimp_diagnostics_results.json" if args.dimp_only else "all_results.json"
    )
    with results_path.open('w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nPhase 1 Complete! Results: {results_path}")


if __name__ == "__main__":
    main()
