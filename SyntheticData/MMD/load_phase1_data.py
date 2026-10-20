"""Load data from Phase 1 for distribution analysis.

This module loads the EXACT same data used in Phase 1 DIMP experiments:
- Real data from real_train.jsonl (QC train pool only, NO test to avoid leakage)
- Qwen synthetic from synthetic_qwen.jsonl (train split)
- Llama synthetic from synthetic_llama.jsonl (train split)

Uses the frozen Phase 1 split IDs.
"""
import json
from pathlib import Path
from typing import Any


# Data paths (same as Phase 1)
QWEN_SYNTHETIC_PATH = Path("../SmaExp/outputs/protocol/synthetic_qwen.jsonl")
LLAMA_SYNTHETIC_PATH = Path("../SmaExp/outputs/protocol/synthetic_llama.jsonl")
REAL_TRAIN_PATH = Path("../SmaExp/outputs/protocol/real_train.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with path.open() as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_phase1_split_ids(splits_dir: Path = Path("../dimp/outputs/phase1/splits")) -> dict[str, list[int]]:
    """Load Phase 1 frozen split IDs."""
    split_ids = {}

    # Real QC train IDs (160 samples, used for QC training)
    with (splits_dir / "real_qc_train_ids.json").open() as f:
        split_ids['real_qc_train'] = json.load(f)

    # Qwen train IDs (320 samples)
    with (splits_dir / "qwen_train_ids.json").open() as f:
        split_ids['qwen_train'] = json.load(f)

    # Llama train IDs (320 samples)
    with (splits_dir / "llama_train_ids.json").open() as f:
        split_ids['llama_train'] = json.load(f)

    return split_ids


def load_real_data(split_ids: dict) -> list[dict]:
    """Load real AG News data (QC train pool only, NO test to avoid leakage)."""
    # Load full real_train.jsonl
    real_full = load_jsonl(REAL_TRAIN_PATH)

    # Filter to QC train IDs only
    qc_train_ids = set(split_ids['real_qc_train'])

    samples = []
    for idx in qc_train_ids:
        item = real_full[idx]
        samples.append({
            'sample_id': f"real_{idx}",
            'source': 'real',
            'text': item['text'],
            'label': item['label'],
        })

    return samples


def load_synthetic_data(source: str, split_ids: dict) -> list[dict]:
    """Load Qwen or Llama synthetic data using Phase 1 train split.

    Args:
        source: 'qwen' or 'llama'
        split_ids: dict with split IDs
    """
    if source == 'qwen':
        data_path = QWEN_SYNTHETIC_PATH
        train_ids = set(split_ids['qwen_train'])
    elif source == 'llama':
        data_path = LLAMA_SYNTHETIC_PATH
        train_ids = set(split_ids['llama_train'])
    else:
        raise ValueError(f"Invalid source: {source}")

    # Load full synthetic data
    full_data = load_jsonl(data_path)

    # Filter to train IDs only
    samples = []
    for idx in train_ids:
        item = full_data[idx]
        samples.append({
            'sample_id': f"{source}_{idx}",
            'source': source,
            'text': item['text'],
            'label': item['label'],
        })

    return samples


def load_all_data() -> dict[str, list[dict]]:
    """Load all three data sources (EXACT same data as Phase 1 DIMP experiments)."""
    print("="*60)
    print("Loading Phase 1 DIMP Data")
    print("="*60)

    print("\nLoading Phase 1 split IDs...")
    split_ids = load_phase1_split_ids()

    print(f"  Real QC train: {len(split_ids['real_qc_train'])} IDs")
    print(f"  Qwen train: {len(split_ids['qwen_train'])} IDs")
    print(f"  Llama train: {len(split_ids['llama_train'])} IDs")

    print("\nLoading Real data (QC train pool only, NO test)...")
    real_data = load_real_data(split_ids)
    print(f"  Loaded {len(real_data)} real samples")

    print("Loading Qwen synthetic data (train split)...")
    qwen_data = load_synthetic_data('qwen', split_ids)
    print(f"  Loaded {len(qwen_data)} Qwen samples")

    print("Loading Llama synthetic data (train split)...")
    llama_data = load_synthetic_data('llama', split_ids)
    print(f"  Loaded {len(llama_data)} Llama samples")

    return {
        'real': real_data,
        'qwen': qwen_data,
        'llama': llama_data,
    }


def split_by_class(data: list[dict]) -> dict[int, list[dict]]:
    """Split data by class label."""
    by_class = {0: [], 1: [], 2: [], 3: []}
    for sample in data:
        by_class[sample['label']].append(sample)
    return by_class


if __name__ == "__main__":
    all_data = load_all_data()

    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)

    for source, data in all_data.items():
        print(f"\n{source.upper()}:")
        by_class = split_by_class(data)
        for label, samples in by_class.items():
            label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            print(f"  {label_names[label]}: {len(samples)} samples")
