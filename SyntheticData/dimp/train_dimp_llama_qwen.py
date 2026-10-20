"""Train classifiers with DIMP-weighted synthetic data and test on real data.

This script implements:
1. Quality Checker training on real data
2. DIMP loss with dynamic sample weighting
3. Four experimental setups:
   - QWEN synthetic (DIMP-weighted) → train QWEN → test on real
   - Llama synthetic (DIMP-weighted) → train Llama → test on real
   - QWEN synthetic → train QWEN → test on Llama real
   - Llama synthetic → train Llama → test on QWEN real

Usage:
  python train_dimp_llama_qwen.py --protocol-dir outputs/protocol_dimp
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# AG News labels
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Data paths - defaults point to generated data in SmaExp, relative to this checkout.
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Save records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_per_class(records: list[dict], n_per_class: int, seed: int = 42) -> list[dict]:
    """Sample n records per class."""
    from collections import defaultdict
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    sampled = []
    for label in sorted(by_label.keys()):
        label_records = by_label[label]
        if len(label_records) < n_per_class:
            raise ValueError(f"Label {label} has only {len(label_records)} records, need {n_per_class}")
        rng.shuffle(label_records)
        sampled.extend(label_records[:n_per_class])

    rng.shuffle(sampled)
    return sampled


class QualityChecker:
    """Quality Checker trained on real data to assess synthetic sample quality."""

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 4):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None

    def train(
        self,
        train_records: list[dict[str, Any]],
        eval_records: list[dict[str, Any]],
        output_dir: Path,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 256,
        save_model: bool = True,
    ) -> dict[str, Any]:
        """Train Quality Checker on real data."""
        print(f"\n=== Training Quality Checker ===")
        print(f"Train samples: {len(train_records)}")
        print(f"Eval samples: {len(eval_records)}")

        # Prepare datasets
        train_ds = Dataset.from_list([
            {"text": row["text"], "labels": int(row["label"])}
            for row in train_records
        ])
        eval_ds = Dataset.from_list([
            {"text": row["text"], "labels": int(row["label"])}
            for row in eval_records
        ])

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
            return self.tokenizer(batch["text"], truncation=True, max_length=max_length)

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        def metrics(eval_pred: Any) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, predictions)),
                "macro_f1": float(f1_score(labels, predictions, average="macro")),
            }

        training_args = TrainingArguments(
            output_dir=str(output_dir / "quality_checker"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch" if save_model else "no",
            save_total_limit=1 if save_model else None,
            load_best_model_at_end=save_model,
            save_only_model=save_model,
            metric_for_best_model="accuracy",
            logging_steps=20,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        # Save model for the full experiment. Smoke tests can disable this to
        # avoid writing large optimizer/model artifacts to a small disk.
        if save_model:
            self.model.save_pretrained(output_dir / "quality_checker")
            self.tokenizer.save_pretrained(output_dir / "quality_checker")

        print(f"Quality Checker trained: accuracy={eval_result['eval_accuracy']:.4f}, "
              f"f1={eval_result['eval_macro_f1']:.4f}")

        return eval_result

    def load(self, model_dir: Path) -> None:
        """Load pre-trained Quality Checker."""
        print(f"Loading Quality Checker from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @torch.no_grad()
    def compute_quality_scores(
        self,
        records: list[dict[str, Any]],
        batch_size: int = 32,
        max_length: int = 256,
    ) -> list[float]:
        """Compute quality scores q_i = P_q(y_i|x_i) for synthetic samples."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Quality Checker not loaded. Call train() or load() first.")

        print(f"\n=== Computing Quality Scores ===")
        print(f"Total samples: {len(records)}")

        self.model.eval()
        device = next(self.model.parameters()).device

        quality_scores = []

        for i in tqdm(range(0, len(records), batch_size), desc="Computing q_i"):
            batch_records = records[i:i+batch_size]
            texts = [r["text"] for r in batch_records]
            labels = torch.tensor([int(r["label"]) for r in batch_records], device=device)

            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Compute P_q(y_i|x_i)
            probs = F.softmax(logits, dim=-1)
            q_scores = probs[range(len(labels)), labels]

            quality_scores.extend(q_scores.cpu().tolist())

        print(f"Quality scores computed: mean={np.mean(quality_scores):.4f}, "
              f"std={np.std(quality_scores):.4f}, "
              f"min={np.min(quality_scores):.4f}, "
              f"max={np.max(quality_scores):.4f}")

        return quality_scores


def compute_dimp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    quality_scores: torch.Tensor,
    epsilon: float = 1e-8,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute DIMP loss with dynamic sample weighting.

    Args:
        logits: Model output logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        quality_scores: Pre-computed q_i from Quality Checker [batch_size]
        epsilon: Small constant to prevent division by zero
        gamma: Power applied to weights (default 1.0 = no transformation)

    Returns:
        loss: DIMP loss scalar
        diagnostics: Dictionary with weight statistics
    """
    batch_size = logits.size(0)

    # Compute per-sample CE loss (reduction='none')
    ce_losses = F.cross_entropy(logits, labels, reduction="none")

    # Compute current model's probability for the given label
    probs = F.softmax(logits, dim=-1)
    p_current = probs[range(batch_size), labels]

    # Compute DIMP weights: w_i = (q_i / p_i) ^ gamma
    # IMPORTANT: detach p_current to prevent gradient flow through weights
    p_current_detached = p_current.detach().clamp(min=epsilon)
    weights = (quality_scores / p_current_detached) ** gamma

    # Check for NaN/Inf
    if torch.isnan(weights).any() or torch.isinf(weights).any():
        print("WARNING: NaN or Inf detected in DIMP weights!")
        print(f"  NaN count: {torch.isnan(weights).sum().item()}")
        print(f"  Inf count: {torch.isinf(weights).sum().item()}")
        weights = torch.nan_to_num(weights, nan=1.0, posinf=10.0, neginf=0.1)

    # Compute weighted loss
    loss = (weights * ce_losses).mean()

    # Diagnostics
    diagnostics = {
        "mean_weight": weights.mean().item(),
        "std_weight": weights.std().item(),
        "min_weight": weights.min().item(),
        "max_weight": weights.max().item(),
        "median_weight": weights.median().item(),
        "mean_quality_score": quality_scores.mean().item(),
        "mean_current_prob": p_current.mean().item(),
    }

    return loss, diagnostics


class DIMPTrainer:
    """Trainer with DIMP loss for synthetic data."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 4,
        loss_type: str = "dimp",
        gamma: float = 1.0,
        epsilon: float = 1e-8,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.gamma = gamma
        self.epsilon = epsilon

    def train(
        self,
        train_records: list[dict[str, Any]],
        quality_scores: list[float],
        test_records: list[dict[str, Any]],
        output_dir: Path,
        run_name: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 256,
    ) -> dict[str, Any]:
        """Train classifier with DIMP loss."""
        print(f"\n=== Training {run_name} with {self.loss_type.upper()} loss ===")
        print(f"Train samples: {len(train_records)}")
        print(f"Test samples: {len(test_records)}")

        # Prepare datasets
        train_data = [
            {
                "text": row["text"],
                "labels": int(row["label"]),
                "quality_score": quality_scores[i],
            }
            for i, row in enumerate(train_records)
        ]

        train_ds = Dataset.from_dict({
            "text": [d["text"] for d in train_data],
            "labels": [d["labels"] for d in train_data],
            "quality_score": [d["quality_score"] for d in train_data],
        })

        test_ds = Dataset.from_list([
            {"text": row["text"], "labels": int(row["label"])}
            for row in test_records
        ])

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        def metrics(eval_pred: Any) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, predictions)),
                "macro_f1": float(f1_score(labels, predictions, average="macro")),
            }

        # Custom trainer for DIMP loss
        class DIMPTrainerClass(Trainer):
            def __init__(self, *args, loss_type="dimp", gamma=1.0, epsilon=1e-8, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_type = loss_type
                self.gamma = gamma
                self.epsilon = epsilon
                self.epoch_diagnostics = []

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                quality_scores = inputs.pop("quality_score", None)

                outputs = model(**inputs)
                logits = outputs.logits

                if self.loss_type == "dimp" and quality_scores is not None:
                    # DIMP loss
                    quality_scores_tensor = torch.tensor(
                        quality_scores, device=logits.device, dtype=logits.dtype
                    )
                    loss, diagnostics = compute_dimp_loss(
                        logits, labels, quality_scores_tensor, self.epsilon, self.gamma
                    )
                    self.epoch_diagnostics.append(diagnostics)
                else:
                    # Standard CE loss
                    loss = F.cross_entropy(logits, labels)

                return (loss, outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir=str(output_dir / run_name),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = DIMPTrainerClass(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=metrics,
            loss_type=self.loss_type,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        # Get predictions and detailed report
        predictions = trainer.predict(test_ds)
        y_pred = predictions.predictions.argmax(-1)
        report = classification_report(
            predictions.label_ids, y_pred, target_names=list(LABELS.values()), digits=4
        )

        # Aggregate diagnostics
        if trainer.epoch_diagnostics:
            avg_diagnostics = {
                key: float(np.mean([d[key] for d in trainer.epoch_diagnostics]))
                for key in trainer.epoch_diagnostics[0].keys()
            }
        else:
            avg_diagnostics = {}

        # Save results
        results = {
            "run": run_name,
            "loss_type": self.loss_type,
            "gamma": self.gamma,
            "train_count": len(train_records),
            "test_count": len(test_records),
            "metrics": eval_result,
            "diagnostics": avg_diagnostics,
        }

        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")

        print(f"Results: accuracy={eval_result['eval_accuracy']:.4f}, "
              f"f1={eval_result['eval_macro_f1']:.4f}")
        if avg_diagnostics:
            print(f"Diagnostics: {json.dumps(avg_diagnostics, indent=2)}")
        print(report)

        return results


def run_experiments(args: argparse.Namespace) -> None:
    """Run all four experiments."""
    seed_everything(args.seed)

    protocol_dir = Path(args.protocol_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    # Load data - THESE PATHS NEED TO BE UPDATED AFTER DATA GENERATION
    print("Loading data...")
    qwen_synthetic = load_jsonl(QWEN_SYNTHETIC_PATH)
    llama_synthetic = load_jsonl(LLAMA_SYNTHETIC_PATH)
    real_train_full = load_jsonl(REAL_TRAIN_PATH)
    real_test = load_jsonl(REAL_TEST_PATH)

    print(f"Loaded: QWEN synthetic={len(qwen_synthetic)}, "
          f"Llama synthetic={len(llama_synthetic)}, "
          f"Real train={len(real_train_full)}, "
          f"Real test={len(real_test)}")

    # Split real data: 80 for Quality Checker training, 20 for QC validation
    # This ensures QC doesn't see test data
    real_qc_train = sample_per_class(real_train_full, args.qc_train_per_class, args.seed)
    remaining_real = [r for r in real_train_full if r not in real_qc_train]
    real_qc_eval = sample_per_class(remaining_real, args.qc_eval_per_class, args.seed + 1)

    print(f"Quality Checker data: train={len(real_qc_train)}, eval={len(real_qc_eval)}")

    # Sample synthetic data: 80 per class for training, 20 per class for testing
    qwen_train = sample_per_class(qwen_synthetic, 80, args.seed)
    qwen_test = sample_per_class(
        [r for r in qwen_synthetic if r not in qwen_train], 20, args.seed + 2
    )

    llama_train = sample_per_class(llama_synthetic, 80, args.seed)
    llama_test = sample_per_class(
        [r for r in llama_synthetic if r not in llama_train], 20, args.seed + 3
    )

    print(f"QWEN: train={len(qwen_train)}, test={len(qwen_test)}")
    print(f"Llama: train={len(llama_train)}, test={len(llama_test)}")

    # Step 1: Train Quality Checker
    qc_dir = protocol_dir / "quality_checker"
    qc = QualityChecker(model_name=args.bert_model, num_labels=4)

    if args.reuse_qc and qc_dir.exists():
        print(f"Reusing existing Quality Checker from {qc_dir}")
        qc.load(qc_dir)
    else:
        qc.train(
            real_qc_train,
            real_qc_eval,
            protocol_dir,
            epochs=args.qc_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    # Step 2: Compute quality scores for synthetic data
    print("\nComputing quality scores for QWEN synthetic data...")
    qwen_quality_scores = qc.compute_quality_scores(qwen_train, batch_size=args.batch_size)

    print("\nComputing quality scores for Llama synthetic data...")
    llama_quality_scores = qc.compute_quality_scores(llama_train, batch_size=args.batch_size)

    # Cache quality scores
    cache_dir = protocol_dir / "quality_scores"
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(
        [{"idx": i, "quality_score": score} for i, score in enumerate(qwen_quality_scores)],
        cache_dir / "qwen_quality_scores.jsonl",
    )
    save_jsonl(
        [{"idx": i, "quality_score": score} for i, score in enumerate(llama_quality_scores)],
        cache_dir / "llama_quality_scores.jsonl",
    )

    # Step 3: Train DIMP models and run experiments
    dimp_trainer = DIMPTrainer(
        model_name=args.bert_model,
        num_labels=4,
        loss_type="dimp",
        gamma=args.gamma,
        epsilon=args.epsilon,
    )

    results = []

    # Experiment 1: QWEN synthetic (DIMP) → test on real
    print("\n" + "="*60)
    print("Experiment 1: QWEN synthetic (DIMP) → test on real")
    print("="*60)
    result1 = dimp_trainer.train(
        qwen_train,
        qwen_quality_scores,
        real_test,
        protocol_dir,
        "exp1_qwen_dimp_test_real",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result1)

    # Experiment 2: Llama synthetic (DIMP) → test on real
    print("\n" + "="*60)
    print("Experiment 2: Llama synthetic (DIMP) → test on real")
    print("="*60)
    result2 = dimp_trainer.train(
        llama_train,
        llama_quality_scores,
        real_test,
        protocol_dir,
        "exp2_llama_dimp_test_real",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result2)

    # Experiment 3: QWEN synthetic (DIMP) → test on Llama test
    print("\n" + "="*60)
    print("Experiment 3: QWEN synthetic (DIMP) → test on Llama synthetic")
    print("="*60)
    result3 = dimp_trainer.train(
        qwen_train,
        qwen_quality_scores,
        llama_test,
        protocol_dir,
        "exp3_qwen_dimp_test_llama",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result3)

    # Experiment 4: Llama synthetic (DIMP) → test on QWEN test
    print("\n" + "="*60)
    print("Experiment 4: Llama synthetic (DIMP) → test on QWEN synthetic")
    print("="*60)
    result4 = dimp_trainer.train(
        llama_train,
        llama_quality_scores,
        qwen_test,
        protocol_dir,
        "exp4_llama_dimp_test_qwen",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result4)

    # Experiment 5: QWEN synthetic (DIMP) → test on QWEN test
    print("\n" + "="*60)
    print("Experiment 5: QWEN synthetic (DIMP) → test on QWEN synthetic")
    print("="*60)
    result5 = dimp_trainer.train(
        qwen_train,
        qwen_quality_scores,
        qwen_test,
        protocol_dir,
        "exp5_qwen_dimp_test_qwen",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result5)

    # Experiment 6: Llama synthetic (DIMP) → test on Llama test
    print("\n" + "="*60)
    print("Experiment 6: Llama synthetic (DIMP) → test on Llama synthetic")
    print("="*60)
    result6 = dimp_trainer.train(
        llama_train,
        llama_quality_scores,
        llama_test,
        protocol_dir,
        "exp6_llama_dimp_test_llama",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    results.append(result6)

    # Save all results
    results_path = protocol_dir / "all_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train classifiers with DIMP-weighted synthetic data"
    )
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default="outputs/protocol_dimp_llama_qwen",
        help="Directory to store all outputs",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="BERT model for classification",
    )
    parser.add_argument(
        "--qc-train-per-class",
        type=int,
        default=40,
        help="Real samples per class for Quality Checker training",
    )
    parser.add_argument(
        "--qc-eval-per-class",
        type=int,
        default=10,
        help="Real samples per class for Quality Checker evaluation",
    )
    parser.add_argument(
        "--qc-epochs",
        type=int,
        default=3,
        help="Training epochs for Quality Checker",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs for DIMP classifiers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma parameter for DIMP weights: w_i = (q_i/p_i)^gamma",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for numerical stability in DIMP weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--reuse-qc",
        action="store_true",
        help="Reuse existing Quality Checker if available",
    )

    args = parser.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
