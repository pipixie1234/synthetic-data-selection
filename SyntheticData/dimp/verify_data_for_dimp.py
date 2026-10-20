"""Verify data format and statistics for DIMP training.

This script checks:
1. Data file existence and format
2. Required fields in each record
3. Label distribution (should have enough samples per class)
4. Data quality metrics

Usage:
  python verify_data_for_dimp.py \
    --qwen-data outputs/protocol/synthetic_qwen.jsonl \
    --llama-data outputs/protocol/synthetic_llama.jsonl \
    --real-train outputs/protocol/real_train.jsonl \
    --real-test outputs/protocol/real_test.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
REQUIRED_FIELDS = ["text", "label"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(encoding="utf-8") as f:
        records = []
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"WARNING: Line {i} is not valid JSON: {e}")
        return records


def verify_record_format(records: list[dict[str, Any]], source: str) -> bool:
    """Verify that records have required fields."""
    print(f"\n=== Verifying {source} ===")

    if not records:
        print(f"  ❌ ERROR: No records found!")
        return False

    print(f"  ✓ Total records: {len(records)}")

    # Check required fields
    missing_fields = defaultdict(int)
    for i, record in enumerate(records):
        for field in REQUIRED_FIELDS:
            if field not in record:
                missing_fields[field] += 1
                if missing_fields[field] <= 3:  # Show first 3 examples
                    print(f"  ⚠ Record {i}: Missing field '{field}'")

    if missing_fields:
        print(f"  ❌ ERROR: {sum(missing_fields.values())} records with missing fields")
        for field, count in missing_fields.items():
            print(f"     - '{field}' missing in {count} records")
        return False
    else:
        print(f"  ✓ All records have required fields: {REQUIRED_FIELDS}")

    # Check label distribution
    label_counts = Counter(record["label"] for record in records)
    print(f"  Label distribution:")
    for label in sorted(LABELS.keys()):
        count = label_counts.get(label, 0)
        label_name = LABELS[label]
        status = "✓" if count >= 100 else "⚠" if count >= 80 else "❌"
        print(f"    {status} {label} ({label_name}): {count} samples")

    min_count = min(label_counts.values()) if label_counts else 0
    if min_count < 80:
        print(f"  ⚠ WARNING: Minimum class size is {min_count}, need at least 100 per class (80 train + 20 test)")
    elif min_count < 100:
        print(f"  ⚠ WARNING: Minimum class size is {min_count}, recommended 100 per class")
    else:
        print(f"  ✓ All classes have sufficient samples (min: {min_count})")

    # Check text quality
    empty_texts = sum(1 for r in records if not str(r.get("text", "")).strip())
    if empty_texts > 0:
        print(f"  ⚠ WARNING: {empty_texts} records have empty text")
    else:
        print(f"  ✓ No empty texts")

    # Text length statistics
    text_lengths = [len(str(r.get("text", "")).split()) for r in records]
    if text_lengths:
        import numpy as np
        print(f"  Text statistics:")
        print(f"    - Mean words: {np.mean(text_lengths):.1f}")
        print(f"    - Median words: {np.median(text_lengths):.1f}")
        print(f"    - Min words: {np.min(text_lengths)}")
        print(f"    - Max words: {np.max(text_lengths)}")

    # Check for duplicates
    texts = [str(r.get("text", "")).strip() for r in records]
    unique_texts = set(texts)
    dup_rate = 1 - len(unique_texts) / len(texts) if texts else 0
    if dup_rate > 0.05:
        print(f"  ⚠ WARNING: {dup_rate*100:.1f}% duplicate texts ({len(texts) - len(unique_texts)} duplicates)")
    else:
        print(f"  ✓ Low duplicate rate: {dup_rate*100:.2f}%")

    return min_count >= 80 and not missing_fields


def check_data_leakage(
    qwen_records: list[dict],
    llama_records: list[dict],
    real_train: list[dict],
    real_test: list[dict],
) -> None:
    """Check for data leakage between datasets."""
    print(f"\n=== Checking Data Leakage ===")

    qwen_texts = set(str(r.get("text", "")).strip() for r in qwen_records)
    llama_texts = set(str(r.get("text", "")).strip() for r in llama_records)
    real_train_texts = set(str(r.get("text", "")).strip() for r in real_train)
    real_test_texts = set(str(r.get("text", "")).strip() for r in real_test)

    # Check real train/test overlap
    overlap_real = real_train_texts & real_test_texts
    if overlap_real:
        print(f"  ❌ ERROR: {len(overlap_real)} texts appear in both real train and test!")
        print(f"     First example: {list(overlap_real)[0][:100]}...")
    else:
        print(f"  ✓ No overlap between real train and test")

    # Check synthetic/real overlap
    qwen_real_overlap = qwen_texts & (real_train_texts | real_test_texts)
    if qwen_real_overlap:
        print(f"  ⚠ WARNING: {len(qwen_real_overlap)} QWEN texts appear in real data")
    else:
        print(f"  ✓ No overlap between QWEN synthetic and real data")

    llama_real_overlap = llama_texts & (real_train_texts | real_test_texts)
    if llama_real_overlap:
        print(f"  ⚠ WARNING: {len(llama_real_overlap)} Llama texts appear in real data")
    else:
        print(f"  ✓ No overlap between Llama synthetic and real data")

    # Check QWEN/Llama overlap
    qwen_llama_overlap = qwen_texts & llama_texts
    if qwen_llama_overlap:
        print(f"  ⚠ INFO: {len(qwen_llama_overlap)} texts appear in both QWEN and Llama")
    else:
        print(f"  ✓ No overlap between QWEN and Llama synthetic data")


def generate_update_instructions(args: argparse.Namespace) -> None:
    """Generate instructions to update the training script."""
    print(f"\n=== Update Instructions ===")
    print(f"Edit train_dimp_llama_qwen.py, lines 43-46:")
    print(f"")
    print(f"QWEN_SYNTHETIC_PATH = \"{args.qwen_data}\"")
    print(f"LLAMA_SYNTHETIC_PATH = \"{args.llama_data}\"")
    print(f"REAL_TRAIN_PATH = \"{args.real_train}\"")
    print(f"REAL_TEST_PATH = \"{args.real_test}\"")
    print(f"")
    print(f"Then run:")
    print(f"  python train_dimp_llama_qwen.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify data for DIMP training")
    parser.add_argument("--qwen-data", type=str, required=True, help="Path to QWEN synthetic data")
    parser.add_argument("--llama-data", type=str, required=True, help="Path to Llama synthetic data")
    parser.add_argument("--real-train", type=str, required=True, help="Path to real train data")
    parser.add_argument("--real-test", type=str, required=True, help="Path to real test data")

    args = parser.parse_args()

    print("="*60)
    print("DIMP Data Verification")
    print("="*60)

    # Load all data
    try:
        qwen_records = load_jsonl(Path(args.qwen_data))
        llama_records = load_jsonl(Path(args.llama_data))
        real_train = load_jsonl(Path(args.real_train))
        real_test = load_jsonl(Path(args.real_test))
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        return 1

    # Verify each dataset
    qwen_ok = verify_record_format(qwen_records, "QWEN synthetic")
    llama_ok = verify_record_format(llama_records, "Llama synthetic")
    real_train_ok = verify_record_format(real_train, "Real train")
    real_test_ok = verify_record_format(real_test, "Real test")

    # Check data leakage
    check_data_leakage(qwen_records, llama_records, real_train, real_test)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    all_ok = qwen_ok and llama_ok and real_train_ok and real_test_ok

    if all_ok:
        print(f"✓ All data files are valid and ready for DIMP training!")
        generate_update_instructions(args)
        return 0
    else:
        print(f"❌ Some data files have issues. Please fix them before training.")
        if not qwen_ok:
            print(f"   - QWEN synthetic data needs fixes")
        if not llama_ok:
            print(f"   - Llama synthetic data needs fixes")
        if not real_train_ok:
            print(f"   - Real train data needs fixes")
        if not real_test_ok:
            print(f"   - Real test data needs fixes")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
