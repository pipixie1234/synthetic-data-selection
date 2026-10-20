"""Enhanced diagnostics collection with per-sample tracking.

This module adds detailed per-sample tracking across epochs for DIMP analysis.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset


def compute_full_sample_diagnostics(
    model,
    tokenizer,
    train_records: list[dict[str, Any]],
    quality_scores: list[float],
    output_path: Path,
    batch_size: int = 32,
    max_length: int = 256,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Compute diagnostics for all training samples at current model state.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        train_records: Training samples
        quality_scores: Pre-computed q_i
        output_path: Path to save CSV
        batch_size: Batch size for inference
        max_length: Max sequence length
        gamma: DIMP gamma parameter
        epsilon: DIMP epsilon parameter

    Returns:
        DataFrame with per-sample diagnostics
    """
    model.eval()
    device = next(model.parameters()).device

    sample_data = []

    with torch.no_grad():
        for i in range(0, len(train_records), batch_size):
            batch_records = train_records[i:i+batch_size]
            batch_q = quality_scores[i:i+batch_size]

            texts = [r['text'] for r in batch_records]
            labels = [r['label'] for r in batch_records]

            # Tokenize
            inputs = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors='pt'
            ).to(device)

            # Forward
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

            # Get p_i (prob of correct label)
            labels_tensor = torch.tensor(labels, device=device)
            p_correct = probs[range(len(labels)), labels].cpu().numpy()

            # Compute weights
            batch_q_array = np.array(batch_q)
            weights = (batch_q_array / np.clip(p_correct, epsilon, None)) ** gamma

            # Get predictions
            preds = logits.argmax(dim=-1).cpu().numpy()

            # Store per-sample info
            for j, record in enumerate(batch_records):
                sample_data.append({
                    'sample_id': i + j,
                    'source': record.get('generator', 'unknown'),
                    'label': record['label'],
                    'text': record['text'][:100],  # Truncate for readability
                    'quality_score': batch_q[j],
                    'p_current': float(p_correct[j]),
                    'weight': float(weights[j]),
                    'prediction': int(preds[j]),
                    'correct': int(preds[j] == labels[j]),
                })

    # Create DataFrame
    df = pd.DataFrame(sample_data)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    return df


def compute_class_weight_statistics(
    df: pd.DataFrame,
    epoch: int,
    seed: int,
    source: str,
) -> pd.DataFrame:
    """Compute per-class weight statistics from sample diagnostics.

    Args:
        df: DataFrame with per-sample diagnostics
        epoch: Current epoch number
        seed: Random seed
        source: Data source (qwen/llama)

    Returns:
        DataFrame with per-class statistics
    """
    class_stats = []

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]

        weights = class_df['weight'].values
        sum_w = weights.sum()
        sum_w2 = (weights ** 2).sum()
        ess = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0

        class_stats.append({
            'source': source,
            'seed': seed,
            'epoch': epoch,
            'class': int(label),
            'num_samples': len(class_df),
            'mean_q': class_df['quality_score'].mean(),
            'mean_p': class_df['p_current'].mean(),
            'mean_weight': weights.mean(),
            'median_weight': float(np.median(weights)),
            'std_weight': weights.std(),
            'min_weight': weights.min(),
            'max_weight': weights.max(),
            'ess': ess,
            'ess_ratio': ess / len(class_df) if len(class_df) > 0 else 0,
        })

    return pd.DataFrame(class_stats)


def extract_top_bottom_samples(
    df: pd.DataFrame,
    n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract top-N and bottom-N weighted samples.

    Args:
        df: DataFrame with per-sample diagnostics
        n: Number of samples to extract

    Returns:
        (top_df, bottom_df)
    """
    df_sorted = df.sort_values('weight', ascending=False)

    top_n = df_sorted.head(n).copy()
    bottom_n = df_sorted.tail(n).copy()

    return top_n, bottom_n


def aggregate_epoch_diagnostics(
    diagnostics_list: list[dict],
) -> pd.DataFrame:
    """Aggregate batch-level diagnostics into epoch-level summary.

    Args:
        diagnostics_list: List of batch diagnostics

    Returns:
        DataFrame with epoch statistics
    """
    if not diagnostics_list:
        return pd.DataFrame()

    # Group by epoch
    from collections import defaultdict
    by_epoch = defaultdict(list)

    for d in diagnostics_list:
        epoch = d.get('epoch', 0)
        by_epoch[epoch].append(d)

    epoch_stats = []
    for epoch, batch_diags in sorted(by_epoch.items()):
        # Aggregate across batches
        stats = {
            'epoch': epoch,
            'num_batches': len(batch_diags),
        }

        # Average numerical fields
        for key in ['mean_q', 'std_q', 'mean_p', 'std_p',
                    'mean_weight', 'std_weight', 'min_weight', 'max_weight', 'median_weight',
                    'ess', 'ess_ratio']:
            values = [d.get(key, 0) for d in batch_diags if key in d]
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))

        # Sum count fields
        for key in ['num_nan', 'num_inf', 'num_weight_gt_10', 'num_weight_gt_20', 'num_weight_gt_50']:
            values = [d.get(key, 0) for d in batch_diags if key in d]
            if values:
                stats[f'{key}_total'] = int(np.sum(values))

        epoch_stats.append(stats)

    return pd.DataFrame(epoch_stats)


def compute_percentiles(weights: np.ndarray) -> dict[str, float]:
    """Compute weight percentiles.

    Args:
        weights: Array of weights

    Returns:
        Dictionary with percentiles
    """
    percentiles = [10, 25, 50, 75, 90, 95]
    result = {}

    for p in percentiles:
        result[f'p{p}'] = float(np.percentile(weights, p))

    return result
