"""Phase 1 trainers with detailed diagnostics.

Implements CE, Quality-only, and DIMP trainers with comprehensive diagnostics collection.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# AG News labels
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


class Phase1Trainer:
    """Base trainer for Phase 1 experiments with detailed diagnostics."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 4,
        loss_type: str = "ce",
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
        test_records: list[dict[str, Any]],
        quality_scores: list[float] | None,
        output_dir: Path,
        run_name: str,
        seed: int = 42,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 256,
        save_checkpoints: bool | None = None,
        checkpoint_dir: Path | None = None,
        additional_eval_records: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Train classifier with specified loss type.

        Args:
            train_records: Training samples
            test_records: Test samples
            quality_scores: Pre-computed q_i from QC (None for CE, required for Quality/DIMP)
            output_dir: Output directory
            run_name: Experiment name
            seed: Random seed
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Max sequence length

        Returns:
            Results dictionary with metrics and diagnostics
        """
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"Loss type: {self.loss_type.upper()}")
        print(f"Train samples: {len(train_records)}")
        print(f"Test samples: {len(test_records)}")
        print(f"{'='*60}\n")

        # Only DIMP needs a persisted model for post-hoc weight diagnostics.
        # CE/Quality runs keep their existing lightweight result artifacts.
        if save_checkpoints is None:
            save_checkpoints = self.loss_type == "dimp"

        model_output_dir = (
            Path(checkpoint_dir) / run_name
            if save_checkpoints and checkpoint_dir is not None
            else output_dir / run_name
        )

        # Prepare datasets
        train_ds = Dataset.from_list([
            {
                "text": row["text"],
                "labels": int(row["label"]),
                # Stable position in the frozen train_records list.  This is
                # used to retrieve q_i after Trainer shuffles each epoch.
                "sample_index": index,
            }
            for index, row in enumerate(train_records)
        ])
        test_ds = Dataset.from_list([
            {"text": row["text"], "labels": int(row["label"])}
            for row in test_records
        ])

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

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

        # Custom trainer with diagnostics
        class DiagnosticsTrainer(Trainer):
            def __init__(self, *args, loss_type, quality_scores, gamma, epsilon, train_records, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_type = loss_type
                self.quality_scores = quality_scores
                self.gamma = gamma
                self.epsilon = epsilon
                self.train_records = train_records
                self.epoch_diagnostics = []
                self.sample_diagnostics = []
                self.current_epoch = 0

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                sample_indices = inputs.pop("sample_index", None)
                outputs = model(**inputs)
                logits = outputs.logits

                # The evaluation set has no training-sample index or q_i.
                # Evaluate it with ordinary CE; sample-weighted loss is only
                # defined for the synthetic training records.
                if sample_indices is None and self.loss_type in ["quality", "dimp"]:
                    loss = F.cross_entropy(logits, labels)
                    diagnostics = self._compute_ce_diagnostics(logits, labels)
                elif self.loss_type == "ce":
                    loss = F.cross_entropy(logits, labels)
                    diagnostics = self._compute_ce_diagnostics(logits, labels)

                elif self.loss_type == "quality":
                    batch_q = self._get_batch_quality_scores(sample_indices, len(labels))
                    loss, diagnostics = self._compute_quality_loss(logits, labels, batch_q)

                elif self.loss_type == "dimp":
                    batch_q = self._get_batch_quality_scores(sample_indices, len(labels))
                    loss, diagnostics = self._compute_dimp_loss(logits, labels, batch_q)

                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")

                if self.state.global_step % 10 == 0:
                    diagnostics['epoch'] = self.state.epoch if self.state.epoch else 0
                    diagnostics['step'] = self.state.global_step
                    self.epoch_diagnostics.append(diagnostics)

                return (loss, outputs) if return_outputs else loss

            def _get_batch_quality_scores(self, sample_indices, batch_size):
                if self.quality_scores is None:
                    raise ValueError("Quality scores required for quality/dimp loss")
                if sample_indices is None:
                    raise ValueError("sample_index is required for quality/dimp diagnostics")
                indices = sample_indices.detach().cpu().tolist()
                batch_q = [self.quality_scores[int(index)] for index in indices]
                return torch.tensor(batch_q, device=next(self.model.parameters()).device, dtype=torch.float32)

            def _compute_ce_diagnostics(self, logits, labels):
                probs = F.softmax(logits, dim=-1)
                p_correct = probs[range(len(labels)), labels]

                return {
                    'mean_p': p_correct.mean().item(),
                    'std_p': p_correct.std().item(),
                    'mean_weight': 1.0,
                    'std_weight': 0.0,
                    'min_weight': 1.0,
                    'max_weight': 1.0,
                    'median_weight': 1.0,
                }

            def _compute_quality_loss(self, logits, labels, quality_scores):
                ce_losses = F.cross_entropy(logits, labels, reduction="none")
                weights = quality_scores.detach()
                loss = (weights * ce_losses).mean()

                probs = F.softmax(logits, dim=-1)
                p_correct = probs[range(len(labels)), labels]

                diagnostics = {
                    'mean_q': quality_scores.mean().item(),
                    'std_q': quality_scores.std().item(),
                    'mean_p': p_correct.mean().item(),
                    'std_p': p_correct.std().item(),
                    'mean_weight': weights.mean().item(),
                    'std_weight': weights.std().item(),
                    'min_weight': weights.min().item(),
                    'max_weight': weights.max().item(),
                    'median_weight': weights.median().item(),
                }

                return loss, diagnostics

            def _compute_dimp_loss(self, logits, labels, quality_scores):
                ce_losses = F.cross_entropy(logits, labels, reduction="none")

                probs = F.softmax(logits, dim=-1)
                p_correct = probs[range(len(labels)), labels]

                p_clamped = p_correct.detach().clamp(min=self.epsilon)
                weights = (quality_scores / p_clamped) ** self.gamma

                num_nan = torch.isnan(weights).sum().item()
                num_inf = torch.isinf(weights).sum().item()
                num_gt_10 = (weights > 10).sum().item()
                num_gt_20 = (weights > 20).sum().item()
                num_gt_50 = (weights > 50).sum().item()

                if num_nan > 0 or num_inf > 0:
                    print(f"\nWARNING: NaN/Inf detected in DIMP weights!")
                    print(f"  NaN: {num_nan}, Inf: {num_inf}")
                    weights = torch.nan_to_num(weights, nan=1.0, posinf=10.0, neginf=0.1)

                loss = (weights * ce_losses).mean()

                sum_w = weights.sum().item()
                sum_w2 = (weights ** 2).sum().item()
                ess = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0
                ess_ratio = ess / len(weights)

                diagnostics = {
                    'mean_q': quality_scores.mean().item(),
                    'std_q': quality_scores.std().item(),
                    'mean_p': p_correct.mean().item(),
                    'std_p': p_correct.std().item(),
                    'mean_weight': weights.mean().item(),
                    'std_weight': weights.std().item(),
                    'min_weight': weights.min().item(),
                    'max_weight': weights.max().item(),
                    'median_weight': weights.median().item(),
                    'ess': ess,
                    'ess_ratio': ess_ratio,
                    'num_nan': num_nan,
                    'num_inf': num_inf,
                    'num_weight_gt_10': num_gt_10,
                    'num_weight_gt_20': num_gt_20,
                    'num_weight_gt_50': num_gt_50,
                }

                return loss, diagnostics

        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_steps=20,
            report_to="none",
            fp16=torch.cuda.is_available(),
            seed=seed,
            # Persisting is handled explicitly after training. This avoids
            # saving optimizer states and multiple large checkpoints.
            save_strategy="no",
            remove_unused_columns=False,
        )

        trainer = DiagnosticsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=metrics,
            loss_type=self.loss_type,
            quality_scores=quality_scores,
            gamma=self.gamma,
            epsilon=self.epsilon,
            train_records=train_records,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        predictions = trainer.predict(test_ds)
        y_pred = predictions.predictions.argmax(-1)
        y_true = predictions.label_ids

        # Handle cases where not all classes appear in test set
        unique_labels = sorted(set(y_true) | set(y_pred))
        target_names_filtered = [LABELS[i] for i in unique_labels]

        report = classification_report(
            y_true, y_pred, labels=unique_labels, target_names=target_names_filtered,
            digits=4, output_dict=True, zero_division=0
        )
        report_text = classification_report(
            y_true, y_pred, labels=unique_labels, target_names=target_names_filtered,
            digits=4, zero_division=0
        )

        # Optional frozen evaluation sets (for example a historical reference
        # split) are evaluated with the same trained model, but do not affect
        # the primary test metrics or training decisions.
        additional_results = {}
        for eval_name, eval_records in (additional_eval_records or {}).items():
            extra_ds = Dataset.from_list([
                {"text": row["text"], "labels": int(row["label"])}
                for row in eval_records
            ])
            extra_ds = extra_ds.map(tokenize, batched=True, remove_columns=["text"])
            extra_prediction = trainer.predict(extra_ds)
            extra_true = extra_prediction.label_ids
            extra_pred = extra_prediction.predictions.argmax(-1)
            extra_labels = sorted(set(extra_true) | set(extra_pred))
            extra_report = classification_report(
                extra_true,
                extra_pred,
                labels=extra_labels,
                target_names=[LABELS[i] for i in extra_labels],
                digits=4,
                output_dict=True,
                zero_division=0,
            )
            additional_results[eval_name] = {
                "metrics": {
                    "accuracy": float(accuracy_score(extra_true, extra_pred)),
                    "macro_f1": float(f1_score(extra_true, extra_pred, average="macro")),
                },
                "per_class_metrics": extra_report,
                "count": len(eval_records),
            }

        if trainer.epoch_diagnostics:
            avg_diagnostics = self._aggregate_diagnostics(trainer.epoch_diagnostics)
        else:
            avg_diagnostics = {}

        results = {
            "run": run_name,
            "loss_type": self.loss_type,
            "gamma": self.gamma,
            "seed": seed,
            "train_count": len(train_records),
            "test_count": len(test_records),
            "metrics": eval_result,
            "per_class_metrics": report,
            "diagnostics": avg_diagnostics,
            "additional_evaluations": additional_results,
        }

        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if save_checkpoints:
            # Keep a stable final model in addition to Trainer's epoch checkpoints.
            final_dir = model_output_dir / "final"
            trainer.save_model(str(final_dir))
            tokenizer.save_pretrained(final_dir)

        (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

        if trainer.epoch_diagnostics:
            diag_path = run_dir / "epoch_diagnostics.json"
            with diag_path.open('w') as f:
                json.dump(trainer.epoch_diagnostics, f, indent=2)

        print(f"\nResults: accuracy={eval_result['eval_accuracy']:.4f}, "
              f"f1={eval_result['eval_macro_f1']:.4f}")

        return results

    def _aggregate_diagnostics(self, epoch_diagnostics):
        if not epoch_diagnostics:
            return {}

        keys = epoch_diagnostics[0].keys()
        aggregated = {}

        for key in keys:
            if key in ['epoch', 'step']:
                continue
            values = [d[key] for d in epoch_diagnostics if key in d]
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                if key.startswith('num_'):
                    aggregated[f'{key}_total'] = int(np.sum(values))

        return aggregated


def train_with_epoch_callbacks(
    trainer_instance,
    train_records: list[dict[str, Any]],
    quality_scores: list[float] | None,
    output_dir: Path,
    run_name: str,
    loss_type: str,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
):
    """Add epoch-end callbacks to collect detailed diagnostics.
    
    This function wraps the training process to collect:
    - Per-sample diagnostics at each epoch end
    - Per-class weight statistics
    - Top/Bottom weighted samples
    """
    from enhanced_diagnostics import (
        compute_full_sample_diagnostics,
        compute_class_weight_statistics,
        extract_top_bottom_samples,
    )
    
    # Create diagnostics directory
    diag_dir = output_dir / run_name / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model and tokenizer from trainer
    model = trainer_instance.model
    tokenizer = trainer_instance.processing_class
    
    # Track across epochs
    all_epoch_samples = []
    all_class_stats = []
    
    # Original training loop hooks
    original_on_epoch_end = getattr(trainer_instance, 'on_epoch_end', None)
    
    def epoch_end_callback(args, state, control, **kwargs):
        """Callback at end of each epoch."""
        if loss_type not in ['quality', 'dimp'] or quality_scores is None:
            return control
        
        current_epoch = int(state.epoch) if state.epoch else 0
        print(f"\n  Collecting diagnostics for epoch {current_epoch}...")
        
        # Compute per-sample diagnostics
        sample_csv_path = diag_dir / f"samples_epoch{current_epoch}.csv"
        df_samples = compute_full_sample_diagnostics(
            model,
            tokenizer,
            train_records,
            quality_scores,
            sample_csv_path,
            batch_size=32,
            max_length=256,
            gamma=gamma,
            epsilon=epsilon,
        )
        
        # Add epoch info
        df_samples['epoch'] = current_epoch
        all_epoch_samples.append(df_samples)
        
        # Compute per-class statistics
        source = 'qwen' if 'qwen' in run_name.lower() else 'llama'
        seed = int(run_name.split('seed')[-1]) if 'seed' in run_name else 42
        
        df_class_stats = compute_class_weight_statistics(
            df_samples, current_epoch, seed, source
        )
        all_class_stats.append(df_class_stats)
        
        # On final epoch, extract top/bottom samples
        if current_epoch == int(args.num_train_epochs) - 1:
            top_samples, bottom_samples = extract_top_bottom_samples(df_samples, n=20)
            top_samples.to_csv(diag_dir / "top20_weights.csv", index=False)
            bottom_samples.to_csv(diag_dir / "bottom20_weights.csv", index=False)
            print(f"  Saved top/bottom 20 weighted samples")
        
        return control
    
    # Add callback to trainer
    from transformers import TrainerCallback
    
    class DiagnosticsCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            return epoch_end_callback(args, state, control, **kwargs)
    
    trainer_instance.add_callback(DiagnosticsCallback())
    
    # Train
    trainer_instance.train()
    
    # Save aggregated diagnostics
    if all_epoch_samples:
        # Combine all epochs
        df_all_samples = pd.concat(all_epoch_samples, ignore_index=True)
        df_all_samples.to_csv(diag_dir / "all_samples_all_epochs.csv", index=False)
    
    if all_class_stats:
        df_all_class_stats = pd.concat(all_class_stats, ignore_index=True)
        df_all_class_stats.to_csv(diag_dir / "class_weight_stats.csv", index=False)
    
    print(f"  Diagnostics saved to: {diag_dir}")
