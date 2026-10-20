"""Generate AG News with Qwen and GPT-Neo, then train/evaluate a BERT classifier.

Examples:
  python synthetic_agnews.py generate --model qwen --num-per-label 1000
  python synthetic_agnews.py generate --model both --num-per-label 1000
  python synthetic_agnews.py train --data outputs/ag_news_synthetic.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_TEXT_TO_ID = {value.lower(): key for key, value in LABELS.items()}
DATASET_NAME = "fancyzhx/ag_news"
_configured_model_root = Path(os.environ["SMAEXP_MODEL_ROOT"]) if "SMAEXP_MODEL_ROOT" in os.environ else None
_default_model_root = Path(__file__).resolve().parents[2] / "cloud" / "model"
MODEL_ROOT = _configured_model_root or _default_model_root


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _records_from_rows(rows: list[dict[str, Any]], source: str, split: str) -> list[dict[str, Any]]:
    return [
        {
            "text": row["text"],
            "label": int(row["label"]),
            "label_name": LABELS[int(row["label"])],
            "source_dataset": source,
            "source_split": split,
            "source_index": int(row["source_index"]),
        }
        for row in rows
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_protocol(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    root = Path(args.protocol_dir)
    root.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATASET_NAME, split="train")
    test = load_dataset(DATASET_NAME, split="test")
    rng = random.Random(args.seed)

    def sample(dataset: Any, per_label: int, split: str) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for label in LABELS:
            indices = [i for i, row in enumerate(dataset) if int(row["label"]) == label]
            if per_label > len(indices):
                raise ValueError(f"{split} class {label} has only {len(indices)} records")
            rng.shuffle(indices)
            for index in indices[:per_label]:
                selected.append({"text": dataset[index]["text"], "label": label, "source_index": index})
        rng.shuffle(selected)
        return _records_from_rows(selected, DATASET_NAME, split)

    real_train = sample(train, args.real_train_per_label, "train")
    real_test = sample(test, args.test_per_label, "test")
    train_texts = {record["text"].strip() for record in real_train}
    test_texts = {record["text"].strip() for record in real_test}
    overlap = train_texts & test_texts
    if overlap:
        raise RuntimeError(f"Real train/test leakage detected: {len(overlap)} exact texts")
    _write_jsonl(root / "real_train.jsonl", real_train)
    _write_jsonl(root / "real_test.jsonl", real_test)
    manifest = {
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "real_train_per_label": args.real_train_per_label,
        "real_test_per_label": args.test_per_label,
        "real_train_count": len(real_train),
        "real_test_count": len(real_test),
        "labels": LABELS,
        "exact_text_overlap": len(overlap),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def build_prompt(label: int, example: str | None = None) -> str:
    seed = f"\nReference style example (do not copy names or numbers): {example}" if example else ""
    return (
        "You are generating a realistic AG News training item.\n"
        f"Target category: {LABELS[label]}\n"
        "Write one original English news article lead of 35-80 words. It must clearly belong to the target category. "
        "Return exactly one line in this format: TEXT: <article>"
        f"{seed}"
    )


def parse_text(raw: str) -> str:
    match = re.search(r"TEXT\s*:\s*(.+)", raw, flags=re.IGNORECASE | re.DOTALL)
    text = match.group(1) if match else raw
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    return text


def load_generator(model_dir: Path, four_bit: bool) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "8bit" in model_dir.name.lower():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return tokenizer, load_legacy_bnb8_model(model_dir, device=device)
    kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
    else:
        kwargs["torch_dtype"] = torch.float32
    # The downloaded GPT-Neo checkpoint stores already-quantized tensors.
    if "8bit" in model_dir.name.lower():
        kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    return tokenizer, model


def load_legacy_bnb8_model(model_dir: Path, device: torch.device | None = None) -> Any:
    """Load the older GPT-Neo blockwise bnb 8-bit checkpoint correctly."""
    from bitsandbytes.functional import dequantize_blockwise
    from transformers import AutoConfig

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    packed = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    state = {}
    for name, value in packed.items():
        if name.endswith(".absmax") or name.endswith(".code"):
            continue
        if name.endswith(".attn.attention.bias") or name.endswith(".attn.attention.masked_bias"):
            continue
        absmax = packed.get(name + ".absmax")
        code = packed.get(name + ".code")
        if value.dtype == torch.uint8 and absmax is not None and code is not None:
            value = dequantize_blockwise(value, absmax=absmax, code=code)
        state[name] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.tie_weights()
    missing = [name for name in missing if name != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(f"GPT-Neo 8-bit state mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
    if target_device.type == "cuda":
        model = model.to(device=target_device, dtype=torch.float16)
    else:
        model = model.to(target_device)
    model.eval()
    return model


@torch.inference_mode()
def generate_one(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int, temperature: float = 1.0) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Custom logits processor to handle inf/nan values in GPT-Neo float16
    from transformers import LogitsProcessorList, LogitsProcessor

    class InfNanLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # Replace inf/nan with large negative values
            scores = torch.where(torch.isinf(scores) | torch.isnan(scores),
                               torch.full_like(scores, -1e9), scores)
            # Clamp extreme values
            scores = torch.clamp(scores, min=-1e9, max=1e9)
            return scores

    logits_processor = LogitsProcessorList([InfNanLogitsProcessor()])

    # Use sampling with higher temperature for diversity
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    except Exception as e:
        # Fallback: try greedy decoding
        print(f"Warning: Sampling failed ({e}), falling back to greedy decoding")
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = output[0, inputs["input_ids"].shape[1] :]
    return parse_text(tokenizer.decode(completion, skip_special_tokens=True))


def generate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    protocol_dir = Path(args.protocol_dir)
    if not (protocol_dir / "real_train.jsonl").exists():
        prepare_protocol(
            argparse.Namespace(
                protocol_dir=str(protocol_dir),
                real_train_per_label=args.real_train_per_label,
                test_per_label=args.test_per_label,
                seed=args.seed,
            )
        )
    real_train = load_jsonl(protocol_dir / "real_train.jsonl")
    forbidden = {record["text"].strip() for record in real_train}
    forbidden.update(record["text"].strip() for record in load_jsonl(protocol_dir / "real_test.jsonl"))
    references = {
        label: [record["text"] for record in real_train if int(record["label"]) == label] for label in LABELS
    }
    model_dirs = []
    if args.model in ("qwen", "both"):
        model_dirs.append(("qwen", Path(args.qwen_dir)))
    if args.model in ("gpt", "both"):
        model_dirs.append(("gpt-neo", Path(args.gpt_dir)))
    if args.model == "llama":
        model_dirs.append(("llama", Path(args.llama_dir)))
    records: list[dict[str, Any]] = []
    for model_name, model_dir in model_dirs:
        tokenizer, model = load_generator(model_dir, args.four_bit)
        for label in LABELS:
            accepted = 0
            attempts = 0
            seen = set()
            progress = tqdm(total=args.num_per_label, desc=f"{model_name}:{LABELS[label]}")
            while accepted < args.num_per_label and attempts < args.max_attempts_per_class:
                attempts += 1
                reference = None
                if args.use_reference and references[label]:
                    reference = references[label][(attempts + args.seed) % len(references[label])]
                # Gradually increase temperature for more diversity
                temperature = 0.9 + (attempts // 50) * 0.1  # 0.9, 1.0, 1.1, 1.2...
                temperature = min(temperature, 1.5)  # Cap at 1.5
                try:
                    text = generate_one(tokenizer, model, build_prompt(label, reference), args.max_new_tokens, temperature)
                except (RuntimeError, torch.cuda.CudaError) as e:
                    print(f"\nGeneration error for {model_name}/{LABELS[label]} attempt {attempts}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                normalized = text.strip()
                word_count = len(normalized.split())
                # Debug: print first few generations
                if attempts <= 3:
                    print(f"\n[Debug] Generated text ({word_count} words): {normalized[:100]}...")
                if word_count < 5 or normalized in forbidden or normalized in seen:
                    if attempts <= 3:
                        print(f"[Debug] Rejected: too short={word_count<5}, duplicate={normalized in forbidden or normalized in seen}")
                    continue
                seen.add(normalized)
                records.append(
                    {
                        "text": normalized,
                        "label": label,
                        "label_name": LABELS[label],
                        "generator": model_name,
                        "source_dataset": "synthetic",
                        "source_split": "generated",
                        "source_index": None,
                    }
                )
                forbidden.add(normalized)
                accepted += 1
                progress.update(1)
            progress.close()
            if accepted < args.num_per_label:
                print(
                    f"WARNING: {model_name}/{LABELS[label]} produced {accepted}/{args.num_per_label}; "
                    f"increase --max-attempts-per-class or inspect the model output."
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = Path(args.output) if args.output else protocol_dir / f"synthetic_{args.model}.jsonl"
    _write_jsonl(output, records)
    print(f"Wrote {len(records)} records to {output}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(record.get("text", "")).strip() for record in records]
    labels = [int(record["label"]) for record in records if "label" in record]
    unique = set(texts)
    token_sets = [set(text.lower().split()) for text in texts if text]
    vocabulary = set().union(*token_sets) if token_sets else set()
    counts = {LABELS[label]: labels.count(label) for label in LABELS}
    return {
        "count": len(texts),
        "unique_count": len(unique),
        "duplicate_rate": round(1 - len(unique) / max(1, len(texts)), 6),
        "empty_rate": round(sum(not text for text in texts) / max(1, len(texts)), 6),
        "mean_words": round(float(np.mean([len(text.split()) for text in texts])) if texts else 0.0, 3),
        "p95_words": round(float(np.percentile([len(text.split()) for text in texts], 95)) if texts else 0.0, 3),
        "vocabulary_size": len(vocabulary),
        "type_token_ratio": round(len(vocabulary) / max(1, sum(len(text.split()) for text in texts)), 6),
        "label_counts": counts,
    }


def evaluate(args: argparse.Namespace) -> None:
    records = load_jsonl(Path(args.data))
    report = dataset_quality(records)
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train_classifier(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    args: argparse.Namespace,
    run_name: str,
) -> dict[str, Any]:
    train_ds = Dataset.from_list([{"text": row["text"], "labels": int(row["label"])} for row in train_records])
    test_ds = Dataset.from_list([{"text": row["text"], "labels": int(row["label"])} for row in test_records])
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, local_files_only=False)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=4)

    def metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
        }

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / run_name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    result = trainer.evaluate()
    predictions = trainer.predict(test_ds)
    y_pred = predictions.predictions.argmax(-1)
    report = classification_report(predictions.label_ids, y_pred, target_names=list(LABELS.values()), digits=4)
    output = {"run": run_name, "train_count": len(train_records), "test_count": len(test_records), "metrics": result}
    print(json.dumps(output, indent=2))
    print(report)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    return output


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    train_records = load_jsonl(Path(args.data))
    test_records = load_jsonl(Path(args.test_data))
    train_classifier(train_records, test_records, args, args.run_name)


def compare(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    root = Path(args.protocol_dir)
    real_train = load_jsonl(root / "real_train.jsonl")
    real_test = load_jsonl(root / "real_test.jsonl")

    # Load synthetic data
    qwen_full = load_jsonl(root / "synthetic_qwen.jsonl")
    llama_full = load_jsonl(root / "synthetic_llama.jsonl")

    if len(qwen_full) != 400 or len(llama_full) != 400:
        raise ValueError(f"Expected 400 samples each, got qwen={len(qwen_full)}, llama={len(llama_full)}")

    # Helper: sample n per class
    def sample_per_class(records: list[dict], n_per_class: int) -> list[dict]:
        from collections import defaultdict
        by_label = defaultdict(list)
        for r in records:
            by_label[r['label']].append(r)
        sampled = []
        for label in sorted(by_label.keys()):
            sampled.extend(by_label[label][:n_per_class])
        return sampled

    # Prepare datasets for 8 experiments
    # Experiment 1-4: Train on different sources, test on real_test
    real_50_train = real_train  # Already 50 per class
    qwen_50_train = sample_per_class(qwen_full, 50)
    llama_50_train = sample_per_class(llama_full, 50)
    mixed_50_train = sample_per_class(qwen_full, 25) + sample_per_class(llama_full, 25)  # 25+25=50 per class

    # Experiment 5-8: Split synthetic data into 80 train / 20 test per class
    qwen_80_train = sample_per_class(qwen_full, 80)
    qwen_20_test = [r for r in qwen_full if r not in qwen_80_train][:80]  # 20 per class

    llama_80_train = sample_per_class(llama_full, 80)
    llama_20_test = [r for r in llama_full if r not in llama_80_train][:80]

    results = []

    # Experiments 1-4: Test on real_test
    print("\n=== Experiments 1-4: Train on different sources, test on real data ===")
    results.append(train_classifier(real_50_train, real_test, args, "exp1_real50_test_real"))
    results.append(train_classifier(qwen_50_train, real_test, args, "exp2_qwen50_test_real"))
    results.append(train_classifier(llama_50_train, real_test, args, "exp3_llama50_test_real"))
    results.append(train_classifier(mixed_50_train, real_test, args, "exp4_mixed50_test_real"))

    # Experiments 5-8: Synthetic train/test
    print("\n=== Experiments 5-8: Synthetic train/test ===")
    results.append(train_classifier(qwen_80_train, qwen_20_test, args, "exp5_qwen80_test_qwen20"))
    results.append(train_classifier(llama_80_train, llama_20_test, args, "exp6_llama80_test_llama20"))
    results.append(train_classifier(qwen_80_train, llama_20_test, args, "exp7_qwen80_test_llama20"))
    results.append(train_classifier(llama_80_train, qwen_20_test, args, "exp8_llama80_test_qwen20"))

    output = Path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== All Results ===")
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", choices=["qwen", "gpt", "llama", "both"], default="both")
    gen.add_argument("--qwen-dir", default=str(MODEL_ROOT / "Qwen3-4B-Instruct-2507"))
    gen.add_argument("--gpt-dir", default=str(MODEL_ROOT / "gpt-neo-2.7B-8bit"))
    gen.add_argument("--llama-dir", default=str(MODEL_ROOT / "Llama-3.2-3B-Instruct"))
    gen.add_argument("--protocol-dir", default="outputs/protocol")
    gen.add_argument("--output", default=None)
    gen.add_argument("--num-per-label", type=int, default=100)
    gen.add_argument("--real-train-per-label", type=int, default=50)
    gen.add_argument("--test-per-label", type=int, default=100)
    gen.add_argument("--max-attempts-per-class", type=int, default=500)
    gen.add_argument("--max-new-tokens", type=int, default=128)
    gen.add_argument("--four-bit", action="store_true")
    gen.add_argument("--use-reference", action="store_true")
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=generate)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol-dir", default="outputs/protocol")
    prep.add_argument("--real-train-per-label", type=int, default=50)
    prep.add_argument("--test-per-label", type=int, default=100)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare_protocol)
    tr = sub.add_parser("train")
    tr.add_argument("--data", default="outputs/protocol/real_train.jsonl")
    tr.add_argument("--test-data", default="outputs/protocol/real_test.jsonl")
    tr.add_argument("--run-name", default="single")
    tr.add_argument("--bert-model", default="bert-base-uncased")
    tr.add_argument("--output-dir", default="outputs/bert_agnews")
    tr.add_argument("--epochs", type=float, default=3)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--learning-rate", type=float, default=2e-5)
    tr.add_argument("--max-length", type=int, default=256)
    tr.add_argument("--seed", type=int, default=42)
    tr.set_defaults(func=train)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--protocol-dir", default="outputs/protocol")
    cmp.add_argument("--bert-model", default="bert-base-uncased")
    cmp.add_argument("--output-dir", default="outputs/comparison_models")
    cmp.add_argument("--results", default="outputs/comparison_results.json")
    cmp.add_argument("--epochs", type=float, default=3)
    cmp.add_argument("--batch-size", type=int, default=16)
    cmp.add_argument("--learning-rate", type=float, default=2e-5)
    cmp.add_argument("--max-length", type=int, default=256)
    cmp.add_argument("--seed", type=int, default=42)
    cmp.set_defaults(func=compare)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", default="outputs/ag_news_synthetic.jsonl")
    ev.add_argument("--output", default="outputs/synthetic_quality.json")
    ev.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
