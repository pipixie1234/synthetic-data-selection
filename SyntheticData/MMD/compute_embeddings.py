"""Compute semantic embeddings using frozen RoBERTa encoder.

Uses roberta-base as frozen encoder to extract semantic representations.
No training, pure inference mode.

Caches embeddings to disk to avoid recomputation.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TextDataset(Dataset):
    """Simple text dataset for batch encoding."""

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def extract_embeddings(
    data: list[dict],
    model_name: str = "bert-base-uncased",
    pooling: str = "cls",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[np.ndarray, list[str], list[int]]:
    """Extract embeddings from text data using frozen encoder.

    Args:
        data: list of dicts with 'sample_id', 'text', 'label'
        model_name: HuggingFace model name
        pooling: 'cls' (CLS token) or 'mean' (mean pooling)
        batch_size: batch size for encoding
        device: cuda or cpu

    Returns:
        embeddings: (N, embed_dim) array
        sample_ids: list of sample IDs
        labels: list of labels
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Don't use DataLoader, process in simple batches to avoid dict collation issues
    all_embeddings = []
    sample_ids = []
    labels = []

    print(f"Extracting embeddings (pooling={pooling})...")
    num_batches = (len(data) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Encoding"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            batch = data[start_idx:end_idx]

            texts = [item['text'] for item in batch]
            batch_ids = [item['sample_id'] for item in batch]
            batch_labels = [item['label'] for item in batch]

            # Tokenize
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            # Forward pass
            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden_dim)

            if pooling == "cls":
                # Use CLS token (first token)
                embeddings = last_hidden[:, 0, :].cpu().numpy()
            elif pooling == "mean":
                # Mean pooling over sequence
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                masked_hidden = last_hidden * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                count = attention_mask.sum(dim=1)
                embeddings = (sum_hidden / count).cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            all_embeddings.append(embeddings)
            sample_ids.extend(batch_ids)
            labels.extend(batch_labels)

    embeddings = np.vstack(all_embeddings)
    print(f"Extracted embeddings: {embeddings.shape}")

    return embeddings, sample_ids, labels


def save_embeddings(
    embeddings: np.ndarray,
    sample_ids: list[str],
    labels: list[int],
    metadata: dict[str, Any],
    output_path: Path
) -> None:
    """Save embeddings and metadata to disk."""
    torch.save({
        'embeddings': embeddings,
        'sample_ids': sample_ids,
        'labels': labels,
        'metadata': metadata,
    }, output_path)
    print(f"Saved embeddings to {output_path}")


def load_embeddings(path: Path) -> dict[str, Any]:
    """Load embeddings from disk."""
    data = torch.load(path, weights_only=False)
    print(f"Loaded embeddings: {data['embeddings'].shape}")
    return data


def compute_and_cache_all_embeddings(
    all_data: dict[str, list[dict]],
    cache_dir: Path = Path("outputs/distribution_analysis/embeddings"),
    model_name: str = "bert-base-uncased",
    pooling: str = "cls",
    force_recompute: bool = False
) -> dict[str, dict[str, Any]]:
    """Compute and cache embeddings for all sources.

    Returns:
        dict mapping source to {embeddings, sample_ids, labels, metadata}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'encoder_name': model_name,
        'pooling_method': pooling,
    }

    results = {}

    for source in ['real', 'qwen', 'llama']:
        cache_path = cache_dir / f"{source}_embeddings.pt"

        if cache_path.exists() and not force_recompute:
            print(f"\nLoading cached embeddings for {source.upper()}...")
            results[source] = load_embeddings(cache_path)
        else:
            print(f"\nComputing embeddings for {source.upper()}...")
            embeddings, sample_ids, labels = extract_embeddings(
                all_data[source],
                model_name=model_name,
                pooling=pooling
            )

            # Add embedding dimension to metadata
            metadata['embedding_dim'] = embeddings.shape[1]

            save_embeddings(embeddings, sample_ids, labels, metadata, cache_path)

            results[source] = {
                'embeddings': embeddings,
                'sample_ids': sample_ids,
                'labels': labels,
                'metadata': metadata,
            }

    # Print metadata
    print("\n" + "="*60)
    print("Embedding Metadata")
    print("="*60)
    for key, value in metadata.items():
        print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    from load_phase1_data import load_all_data

    all_data = load_all_data()
    embeddings_data = compute_and_cache_all_embeddings(all_data)

    print("\n" + "="*60)
    print("Embeddings Summary")
    print("="*60)
    for source, data in embeddings_data.items():
        print(f"{source.upper()}: {data['embeddings'].shape}")
