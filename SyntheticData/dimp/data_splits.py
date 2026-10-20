"""Fixed data splits management for Phase 1 experiments.

This module ensures all experiments use identical train/test splits for fair comparison.
"""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def create_fixed_splits(
    qwen_full: list[dict[str, Any]],
    llama_full: list[dict[str, Any]],
    real_train_full: list[dict[str, Any]],
    real_test_full: list[dict[str, Any]],
    seed: int = 42,
    qc_train_per_class: int = 40,
    qc_eval_per_class: int = 10,
    synthetic_train_per_class: int = 80,
    synthetic_test_per_class: int = 20,
) -> dict[str, Any]:
    """Create fixed data splits and save IDs.

    Returns:
        Dictionary with all split IDs and records
    """
    rng = random.Random(seed)

    # Helper: group by label
    def group_by_label(records):
        by_label = defaultdict(list)
        for i, r in enumerate(records):
            by_label[r['label']].append(i)
        return by_label

    # Split Qwen: 80 train + 20 test per class
    qwen_by_label = group_by_label(qwen_full)
    qwen_train_ids = []
    qwen_test_ids = []

    for label in sorted(qwen_by_label.keys()):
        indices = qwen_by_label[label].copy()
        rng.shuffle(indices)
        qwen_train_ids.extend(indices[:synthetic_train_per_class])
        qwen_test_ids.extend(indices[synthetic_train_per_class:synthetic_train_per_class + synthetic_test_per_class])

    # Split Llama: 80 train + 20 test per class
    llama_by_label = group_by_label(llama_full)
    llama_train_ids = []
    llama_test_ids = []

    for label in sorted(llama_by_label.keys()):
        indices = llama_by_label[label].copy()
        rng.shuffle(indices)
        llama_train_ids.extend(indices[:synthetic_train_per_class])
        llama_test_ids.extend(indices[synthetic_train_per_class:synthetic_train_per_class + synthetic_test_per_class])

    # Split real train: 40 QC train + 10 QC eval per class
    real_by_label = group_by_label(real_train_full)
    real_qc_train_ids = []
    real_qc_eval_ids = []

    for label in sorted(real_by_label.keys()):
        indices = real_by_label[label].copy()
        rng.shuffle(indices)
        real_qc_train_ids.extend(indices[:qc_train_per_class])
        real_qc_eval_ids.extend(indices[qc_train_per_class:qc_train_per_class + qc_eval_per_class])

    # Real test IDs (all of them, no splitting needed)
    real_test_ids = list(range(len(real_test_full)))

    # Create splits dictionary
    splits = {
        'seed': seed,
        'qwen_train_ids': qwen_train_ids,
        'qwen_test_ids': qwen_test_ids,
        'llama_train_ids': llama_train_ids,
        'llama_test_ids': llama_test_ids,
        'real_qc_train_ids': real_qc_train_ids,
        'real_qc_eval_ids': real_qc_eval_ids,
        'real_test_ids': real_test_ids,
        'config': {
            'qc_train_per_class': qc_train_per_class,
            'qc_eval_per_class': qc_eval_per_class,
            'synthetic_train_per_class': synthetic_train_per_class,
            'synthetic_test_per_class': synthetic_test_per_class,
        }
    }

    return splits


def save_splits(splits: dict[str, Any], output_dir: Path) -> None:
    """Save split IDs to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual ID files
    for key in ['qwen_train_ids', 'qwen_test_ids', 'llama_train_ids', 'llama_test_ids',
                'real_qc_train_ids', 'real_qc_eval_ids', 'real_test_ids']:
        path = output_dir / f"{key}.json"
        with path.open('w') as f:
            json.dump(splits[key], f, indent=2)

    # Save complete splits metadata
    metadata_path = output_dir / "splits_metadata.json"
    with metadata_path.open('w') as f:
        json.dump({
            'seed': splits['seed'],
            'config': splits['config'],
            'counts': {
                'qwen_train': len(splits['qwen_train_ids']),
                'qwen_test': len(splits['qwen_test_ids']),
                'llama_train': len(splits['llama_train_ids']),
                'llama_test': len(splits['llama_test_ids']),
                'real_qc_train': len(splits['real_qc_train_ids']),
                'real_qc_eval': len(splits['real_qc_eval_ids']),
                'real_test': len(splits['real_test_ids']),
            }
        }, f, indent=2)

    print(f"Splits saved to: {output_dir}")


def load_splits(splits_dir: Path) -> dict[str, Any]:
    """Load splits from saved JSON files."""
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    splits = {}

    # Load individual ID files
    for key in ['qwen_train_ids', 'qwen_test_ids', 'llama_train_ids', 'llama_test_ids',
                'real_qc_train_ids', 'real_qc_eval_ids', 'real_test_ids']:
        path = splits_dir / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        with path.open() as f:
            splits[key] = json.load(f)

    # Load metadata
    metadata_path = splits_dir / "splits_metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)
            splits['seed'] = metadata['seed']
            splits['config'] = metadata['config']

    print(f"Splits loaded from: {splits_dir}")
    return splits


def apply_splits(
    qwen_full: list[dict[str, Any]],
    llama_full: list[dict[str, Any]],
    real_train_full: list[dict[str, Any]],
    real_test_full: list[dict[str, Any]],
    splits: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Apply splits to get actual data records.

    Returns:
        Dictionary with split datasets
    """
    datasets = {
        'qwen_train': [qwen_full[i] for i in splits['qwen_train_ids']],
        'qwen_test': [qwen_full[i] for i in splits['qwen_test_ids']],
        'llama_train': [llama_full[i] for i in splits['llama_train_ids']],
        'llama_test': [llama_full[i] for i in splits['llama_test_ids']],
        'real_qc_train': [real_train_full[i] for i in splits['real_qc_train_ids']],
        'real_qc_eval': [real_train_full[i] for i in splits['real_qc_eval_ids']],
        'real_test': [real_test_full[i] for i in splits['real_test_ids']],
        'real_train_full': real_train_full,  # Keep full for Real200 baseline
    }

    return datasets


def verify_splits_no_leakage(splits: dict[str, Any]) -> dict[str, bool]:
    """Verify no data leakage between splits.

    Returns:
        Dictionary of check results
    """
    checks = {}

    # Check 1: QC train and QC eval should not overlap
    qc_train_set = set(splits['real_qc_train_ids'])
    qc_eval_set = set(splits['real_qc_eval_ids'])
    checks['qc_train_eval_disjoint'] = len(qc_train_set & qc_eval_set) == 0

    # Check 2: Qwen train and test should not overlap
    qwen_train_set = set(splits['qwen_train_ids'])
    qwen_test_set = set(splits['qwen_test_ids'])
    checks['qwen_train_test_disjoint'] = len(qwen_train_set & qwen_test_set) == 0

    # Check 3: Llama train and test should not overlap
    llama_train_set = set(splits['llama_train_ids'])
    llama_test_set = set(splits['llama_test_ids'])
    checks['llama_train_test_disjoint'] = len(llama_train_set & llama_test_set) == 0

    # Check 4: All IDs should be valid indices
    checks['all_ids_non_negative'] = all(
        all(i >= 0 for i in splits[key])
        for key in ['qwen_train_ids', 'qwen_test_ids', 'llama_train_ids',
                    'llama_test_ids', 'real_qc_train_ids', 'real_qc_eval_ids']
    )

    checks['all_passed'] = all(checks.values())

    return checks
