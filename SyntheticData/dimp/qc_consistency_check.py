"""Quality Checker save/load consistency verification.

This module checks whether QC checkpoint saving and loading preserves model predictions.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def check_qc_consistency(
    qc,
    eval_records: list[dict[str, Any]],
    save_dir: Path,
    batch_size: int = 32,
    max_length: int = 256,
) -> dict[str, Any]:
    """Check QC save/load consistency.

    Args:
        qc: Trained QualityChecker instance
        eval_records: Records to test on
        save_dir: Directory where QC is saved
        batch_size: Batch size for inference
        max_length: Max sequence length

    Returns:
        Dictionary with consistency check results
    """
    print("\n" + "="*60)
    print("Quality Checker Save/Load Consistency Check")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Get predictions BEFORE reload
    print("\nStep 1: Computing predictions before reload...")
    logits_before, probs_before, preds_before = _compute_predictions(
        qc, eval_records, device, batch_size, max_length
    )

    # Step 2: Save QC (already saved, but verify path exists)
    if not save_dir.exists():
        raise FileNotFoundError(f"QC save directory not found: {save_dir}")
    print(f"\nStep 2: QC checkpoint exists at: {save_dir}")

    # Step 3: Reload QC from checkpoint
    print("\nStep 3: Reloading QC from checkpoint...")
    qc_reloaded = _reload_qc(save_dir, device)

    # Step 4: Get predictions AFTER reload
    print("\nStep 4: Computing predictions after reload...")
    logits_after, probs_after, preds_after = _compute_predictions(
        qc_reloaded, eval_records, device, batch_size, max_length
    )

    # Step 5: Compare
    print("\nStep 5: Comparing predictions...")
    results = _compare_predictions(
        logits_before, logits_after,
        probs_before, probs_after,
        preds_before, preds_after
    )

    # Step 6: Determine pass/fail
    results['passed'] = (
        results['max_abs_logit_diff'] < 1e-5 and
        results['prediction_match_rate'] == 1.0
    )

    print("\n" + "="*60)
    print("Consistency Check Results:")
    print("="*60)
    print(f"Max absolute logit difference: {results['max_abs_logit_diff']:.2e}")
    print(f"Mean absolute logit difference: {results['mean_abs_logit_diff']:.2e}")
    print(f"Max absolute prob difference: {results['max_abs_prob_diff']:.2e}")
    print(f"Prediction match rate: {results['prediction_match_rate']:.4f}")
    print(f"Status: {'PASSED' if results['passed'] else 'FAILED'}")
    print("="*60)

    if not results['passed']:
        print("\nWARNING: Consistency check FAILED!")
        print("QC checkpoint loading may have issues.")
        print("Please investigate before running experiments.")

    return results


def _compute_predictions(qc, records, device, batch_size, max_length):
    """Compute logits, probs, and predictions for given records."""
    qc.model.eval()
    qc.model.to(device)

    all_logits = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i+batch_size]
            texts = [r['text'] for r in batch_records]

            # Tokenize
            inputs = qc.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors='pt'
            ).to(device)

            # Forward pass
            outputs = qc.model(**inputs)
            logits = outputs.logits.cpu()
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_logits.append(logits)
            all_probs.append(probs)
            all_preds.append(preds)

    # Concatenate
    logits = torch.cat(all_logits, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()

    return logits, probs, preds


def _reload_qc(save_dir, device):
    """Reload QC from checkpoint."""
    class ReloadedQC:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    model.eval()
    model.to(device)

    return ReloadedQC(model, tokenizer)


def _compare_predictions(logits_before, logits_after, probs_before, probs_after, preds_before, preds_after):
    """Compare predictions before and after reload."""
    # Logit differences
    logit_diff = np.abs(logits_before - logits_after)
    max_abs_logit_diff = np.max(logit_diff)
    mean_abs_logit_diff = np.mean(logit_diff)

    # Prob differences
    prob_diff = np.abs(probs_before - probs_after)
    max_abs_prob_diff = np.max(prob_diff)
    mean_abs_prob_diff = np.mean(prob_diff)

    # Prediction match rate
    prediction_match = (preds_before == preds_after).astype(float)
    prediction_match_rate = np.mean(prediction_match)

    results = {
        'max_abs_logit_diff': float(max_abs_logit_diff),
        'mean_abs_logit_diff': float(mean_abs_logit_diff),
        'max_abs_prob_diff': float(max_abs_prob_diff),
        'mean_abs_prob_diff': float(mean_abs_prob_diff),
        'prediction_match_rate': float(prediction_match_rate),
        'num_samples': len(preds_before),
        'num_mismatches': int(np.sum(preds_before != preds_after)),
    }

    return results


def save_consistency_check_results(results: dict[str, Any], output_path: Path) -> None:
    """Save consistency check results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\nConsistency check results saved to: {output_path}")
