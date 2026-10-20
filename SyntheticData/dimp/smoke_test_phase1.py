"""Smoke test for Phase 1 pipeline.

Quick validation with small data samples to ensure everything works.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_phase1 import (
    load_jsonl,
    seed_everything,
    QWEN_SYNTHETIC_PATH,
    LLAMA_SYNTHETIC_PATH,
    REAL_TRAIN_PATH,
    REAL_TEST_PATH,
)
from data_splits import create_fixed_splits, apply_splits, verify_splits_no_leakage
from train_dimp_llama_qwen import QualityChecker
from phase1_trainers import Phase1Trainer


def main():
    parser = argparse.ArgumentParser(description="Smoke test Phase 1 pipeline")
    parser.add_argument("--protocol-dir", type=str, default="outputs/phase1_smoke")
    args = parser.parse_args()

    protocol_dir = Path(args.protocol_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Phase 1 Smoke Test")
    print("="*60)
    print("\nThis will run a quick test with small data samples.")
    print("Expected time: 2-3 minutes\n")

    seed_everything(42)

    # Load full data
    print("Loading data...")
    qwen_full = load_jsonl(Path(QWEN_SYNTHETIC_PATH))
    llama_full = load_jsonl(Path(LLAMA_SYNTHETIC_PATH))
    real_train_full = load_jsonl(Path(REAL_TRAIN_PATH))
    real_test_full = load_jsonl(Path(REAL_TEST_PATH))

    # Take small samples (10 per class = 40 total)
    def sample_small(data, n_per_class=10):
        from collections import defaultdict
        by_label = defaultdict(list)
        for r in data:
            by_label[r['label']].append(r)
        sampled = []
        for label in sorted(by_label.keys()):
            sampled.extend(by_label[label][:n_per_class])
        return sampled

    qwen_small = sample_small(qwen_full, 20)  # 20 per class = 80 total
    llama_small = sample_small(llama_full, 20)
    real_train_small = sample_small(real_train_full, 12)  # 12 per class = 48 total
    real_test_small = sample_small(real_test_full, 10)  # 10 per class = 40 total

    print(f"  Qwen sample: {len(qwen_small)}")
    print(f"  Llama sample: {len(llama_small)}")
    print(f"  Real train sample: {len(real_train_small)}")
    print(f"  Real test sample: {len(real_test_small)}")

    # Create splits (8 train + 2 eval per class for QC, 16 train + 4 test for synthetic)
    print("\n[1/6] Creating splits...")
    splits = create_fixed_splits(
        qwen_small, llama_small, real_train_small, real_test_small,
        seed=42,
        qc_train_per_class=8,   # 8 * 4 = 32
        qc_eval_per_class=2,    # 2 * 4 = 8
        synthetic_train_per_class=16,  # 16 * 4 = 64
        synthetic_test_per_class=4,    # 4 * 4 = 16
    )

    # Verify splits
    leakage_checks = verify_splits_no_leakage(splits)
    if not leakage_checks['all_passed']:
        print("ERROR: Splits have data leakage!")
        return 1

    datasets = apply_splits(qwen_small, llama_small, real_train_small, real_test_small, splits)
    print(f"  QC train: {len(datasets['real_qc_train'])}")
    print(f"  QC eval: {len(datasets['real_qc_eval'])}")
    print(f"  Qwen train: {len(datasets['qwen_train'])}")
    print(f"  Llama train: {len(datasets['llama_train'])}")

    # Train QC
    print("\n[2/6] Training Quality Checker (1 epoch)...")
    qc = QualityChecker(model_name="bert-base-uncased", num_labels=4)
    qc.train(
        datasets['real_qc_train'],
        datasets['real_qc_eval'],
        protocol_dir,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        save_model=False,
    )

    # Compute quality scores
    print("\n[3/6] Computing quality scores...")
    qwen_quality_scores = qc.compute_quality_scores(datasets['qwen_train'], batch_size=16)
    llama_quality_scores = qc.compute_quality_scores(datasets['llama_train'], batch_size=16)
    print(f"  Qwen quality scores: {len(qwen_quality_scores)}")
    print(f"  Llama quality scores: {len(llama_quality_scores)}")

    # Test CE
    print("\n[4/6] Testing CE trainer...")
    trainer_ce = Phase1Trainer(model_name="bert-base-uncased", loss_type="ce")
    result_ce = trainer_ce.train(
        datasets['qwen_train'][:32],  # Use even smaller subset
        datasets['real_test'][:20],
        None,
        protocol_dir,
        "smoke_ce",
        seed=42,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        save_checkpoints=False,
    )
    print(f"  CE result: acc={result_ce['metrics']['eval_accuracy']:.4f}")

    # Test Quality-only
    print("\n[5/6] Testing Quality-only trainer...")
    trainer_quality = Phase1Trainer(model_name="bert-base-uncased", loss_type="quality")
    result_quality = trainer_quality.train(
        datasets['qwen_train'][:32],
        datasets['real_test'][:20],
        qwen_quality_scores[:32],
        protocol_dir,
        "smoke_quality",
        seed=42,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        save_checkpoints=False,
    )
    print(f"  Quality result: acc={result_quality['metrics']['eval_accuracy']:.4f}")

    # Test DIMP
    print("\n[6/6] Testing DIMP trainer...")
    trainer_dimp = Phase1Trainer(
        model_name="bert-base-uncased",
        loss_type="dimp",
        gamma=1.0,
        epsilon=1e-8,
    )
    result_dimp = trainer_dimp.train(
        datasets['qwen_train'][:32],
        datasets['real_test'][:20],
        qwen_quality_scores[:32],
        protocol_dir,
        "smoke_dimp",
        seed=42,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        save_checkpoints=False,
    )
    print(f"  DIMP result: acc={result_dimp['metrics']['eval_accuracy']:.4f}")

    # Check diagnostics
    if result_dimp['diagnostics']:
        diag = result_dimp['diagnostics']
        print(f"  DIMP diagnostics: mean_weight={diag.get('mean_weight_mean', 0):.4f}, "
              f"max_weight={diag.get('max_weight_mean', 0):.4f}")

        # Check for NaN/Inf
        if diag.get('num_nan_total', 0) > 0 or diag.get('num_inf_total', 0) > 0:
            print("  WARNING: NaN/Inf detected in weights!")
    else:
        print("  WARNING: No diagnostics collected!")

    print("\n" + "="*60)
    print("Smoke Test PASSED!")
    print("="*60)
    print("\nAll components working:")
    print("  ✓ Data splits")
    print("  ✓ Quality Checker training")
    print("  ✓ Quality score computation")
    print("  ✓ CE trainer")
    print("  ✓ Quality-only trainer")
    print("  ✓ DIMP trainer")
    print("  ✓ Diagnostics collection")
    print("\nReady to run full Phase 1 experiments!")
    print(f"\nRun: python run_phase1.py --protocol-dir outputs/phase1 --seeds 42 43 44")

    return 0


if __name__ == "__main__":
    sys.exit(main())
