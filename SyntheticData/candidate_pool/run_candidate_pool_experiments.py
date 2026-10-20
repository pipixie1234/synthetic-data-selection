"""Expand frozen Qwen/Llama candidate pools and evaluate nested TopK pools.

The experiment deliberately keeps the frozen Real-Anchored TopK scorer and the
Qwen40+Llama40 training allocation.  Generation is incremental and resumable;
analysis never looks at the final holdout.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "SyntheticData"
PROTOCOL = SYN / "SmaExp" / "outputs" / "protocol"
SPLITS = SYN / "dimp" / "outputs" / "phase1" / "splits"
FROZEN = SYN / "exp" / "outputs" / "curation_validation"
CLOUD_DATA = ROOT / "cloud" / "model" / "extra_data"
DEFAULT_OUT = Path(os.environ.get("CANDIDATE_POOL_OUTPUT_DIR", str(CLOUD_DATA / "candidate_pool" / "outputs"))).resolve()
OUT = DEFAULT_OUT
LEGACY_OUT = SYN / "candidate_pool" / "outputs"
MIN_FREE_BYTES = 2 * 1024**3

# Keep Hugging Face, torch and Triton caches off the small workspace overlay.
_cache_root = CLOUD_DATA / "candidate_pool" / "cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache_root / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_cache_root / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_root / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(_cache_root / "torch"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_cache_root / "triton"))
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}
POOL_SIZES = [80, 160, 250, 500]
MODEL_SEEDS = [42, 43, 44]
EXPANSION_SEED = 2027
PROMPT_VERSION = "synthetic_agnews.build_prompt.v1"

sys.path.insert(0, str(SYN / "exp"))
sys.path.insert(0, str(SYN / "SmaExp"))
import run_distribution_curation as curation  # noqa: E402
import run_e1_e2_validation as validation  # noqa: E402
import synthetic_agnews as generator  # noqa: E402


def set_output_dir(path: str | Path | None) -> None:
    global OUT
    if path:
        OUT = Path(path).expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)


def ensure_disk_budget(required_bytes: int = MIN_FREE_BYTES) -> None:
    usage = shutil.disk_usage(OUT)
    if usage.free < required_bytes:
        raise RuntimeError(
            f"Insufficient free space on filesystem for {OUT}: "
            f"{usage.free / 1024**3:.2f} GiB free, "
            f"need at least {required_bytes / 1024**3:.2f} GiB."
        )


def cuda_preflight() -> None:
    """Check CUDA visibility without allocating a probe tensor.

    HAMI can reject a tiny ``torch.zeros(..., device='cuda')`` allocation in
    an otherwise healthy process with ``CUDA driver error: invalid argument``.
    The actual trainer performs the real model transfer, so preflight should
    only verify device enumeration and leave allocation to ``Phase1Trainer``.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the GPU allocation and CUDA_VISIBLE_DEVICES.")
    try:
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        print(f"CUDA device visible: {device} ({name}); skipping probe allocation for HAMI compatibility")
    except Exception as exc:
        raise RuntimeError(
            "CUDA device enumeration failed before BERT training. Start the command with "
            "LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES=0. "
            f"Original error: {exc}"
        ) from exc


def migrate_legacy_partial_outputs() -> None:
    """Copy only small resumable generation artifacts from the old location."""
    if OUT == LEGACY_OUT or (OUT / "generation").exists():
        return
    legacy_generation = LEGACY_OUT / "generation"
    if not legacy_generation.exists():
        return
    target = OUT / "generation"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("qwen_new.jsonl", "llama_new.jsonl", "global_text_registry.txt", "global_text_registry.lock"):
        src = legacy_generation / name
        dst = target / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    manifest = LEGACY_OUT / "generation_manifest.json"
    if manifest.exists() and not (OUT / manifest.name).exists():
        shutil.copy2(manifest, OUT / manifest.name)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def frozen_data() -> dict[str, list[dict[str, Any]]]:
    return curation.load_frozen_data()


def old_candidates(data: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for source in ("qwen", "llama"):
        for row in data[f"{source}_train"]:
            rows.append({
                "sample_id": row["sample_id"], "source": source,
                "class": LABEL_NAMES[int(row["label"])], "label": int(row["label"]),
                "text": str(row["text"]), "is_legacy_candidate": True,
                "generation_seed": "", "generation_batch": "", "prompt_version": "legacy",
                "model_checkpoint": "legacy_generation_checkpoint_unknown",
            })
    return rows


def generation_manifest(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers
    paths = {"qwen": str(Path(args.qwen_dir).resolve()), "llama": str(Path(args.llama_dir).resolve())}
    configs = {}
    for source, path in paths.items():
        cfg = Path(path) / "config.json"
        configs[source] = json.loads(cfg.read_text()) if cfg.exists() else {"status": "UNKNOWN_OR_CHANGED"}
    return {
        "experiment": "expanded_candidate_pool",
        "candidate_target_per_source_class": 500,
        "legacy_candidate_policy": "reuse frozen qwen_train/llama_train only; synthetic test excluded",
        "new_candidates_per_source_class": 420,
        "model_paths": paths,
        "model_names": {"qwen": "Qwen3-4B-Instruct-2507", "llama": "Llama-3.2-3B-Instruct"},
        "checkpoint_configs": configs,
        "tokenizer_paths": paths,
        "prompt_version": PROMPT_VERSION,
        "prompt_template": "synthetic_agnews.build_prompt; target category; one 35-80 word lead; TEXT prefix",
        "generation_config": {"max_new_tokens": 128, "do_sample": True, "top_p": 0.95, "top_k": 50,
                               "temperature_schedule": "0.9 + floor(attempt/50)*0.1, capped at 1.5",
                               "use_reference": False, "four_bit": bool(getattr(args, "four_bit", False)),
                               "max_attempts_per_class": int(getattr(args, "max_attempts_per_class", 0)),
                               "batch_size": int(getattr(args, "batch_size", 0))},
        "historical_invocation_args": "UNKNOWN_OR_CHANGED: old files did not record exact seed/attempt metadata",
        "new_generation_seed_base": 2027000,
        "expansion_pool_seed": EXPANSION_SEED,
        "packages": {"torch": torch.__version__, "transformers": transformers.__version__},
        "final_holdout_accessed": False,
    }


def seed_for(source: str, label: int, attempt: int) -> int:
    source_offset = 0 if source == "qwen" else 1000000
    return 2027000 + source_offset + label * 100000 + attempt


def generate_source(args: argparse.Namespace) -> None:
    set_output_dir(args.data_dir)
    migrate_legacy_partial_outputs()
    ensure_disk_budget()
    source = args.source
    manifest_path = OUT / "generation_manifest.json"
    manifest_lock_path = OUT / "generation" / "generation_manifest.lock"
    manifest_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = generation_manifest(args)
        else:
            manifest = generation_manifest(args)
        manifest.setdefault("generated_sources", [])
        if source not in manifest["generated_sources"]:
            manifest["generated_sources"].append(source)
        manifest["storage_output_dir"] = str(OUT)
        manifest["storage_free_bytes_at_start"] = shutil.disk_usage(OUT).free
        atomic_json_write(manifest_path, manifest)
        fcntl.flock(lock, fcntl.LOCK_UN)
    data = frozen_data()
    legacy = old_candidates(data)
    forbidden = {norm_text(x["text"]) for x in legacy}
    for path in (PROTOCOL / "real_train.jsonl", FROZEN / "dev2000.csv"):
        if path.suffix == ".jsonl":
            forbidden.update(norm_text(x["text"]) for x in jsonl_read(path))
        elif path.exists():
            forbidden.update(norm_text(x) for x in pd.read_csv(path)["text"].tolist())
    shard = OUT / "generation" / f"{source}_new.jsonl"
    lock_path = OUT / "generation" / "global_text_registry.lock"
    registry_path = OUT / "generation" / "global_text_registry.txt"
    shard.parent.mkdir(parents=True, exist_ok=True)
    existing = jsonl_read(shard)
    counts = collections.Counter(int(x["label"]) for x in existing)
    # Seed the cross-process registry once with all old and already generated text.
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry_path.touch()
        current = {x.strip() for x in registry_path.read_text(encoding="utf-8").splitlines() if x.strip()}
        missing = forbidden | {norm_text(x["text"]) for x in existing}
        with registry_path.open("a", encoding="utf-8") as reg:
            for value in sorted(missing - current):
                reg.write(value + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
    model_dir = Path(args.qwen_dir if source == "qwen" else args.llama_dir)
    tokenizer, model = generator.load_generator(model_dir, args.four_bit)
    references = {label: [x["text"] for x in jsonl_read(PROTOCOL / "real_train.jsonl") if int(x["label"]) == label] for label in LABEL_NAMES}
    try:
        for label in LABEL_NAMES:
            attempt = 0
            while counts[label] < args.new_per_class and attempt < args.max_attempts_per_class:
                attempt += 1
                sample_seed = seed_for(source, label, attempt)
                generator.seed_everything(sample_seed)
                temperature = min(0.9 + (attempt // 50) * 0.1, 1.5)
                try:
                    text = generator.generate_one(tokenizer, model, generator.build_prompt(label), args.max_new_tokens, temperature)
                except Exception as exc:
                    print(f"{source}/{LABEL_NAMES[label]} attempt={attempt} generation error: {exc}", flush=True)
                    continue
                cleaned = re.sub(r"\s+", " ", str(text).strip().strip('"'))
                if len(cleaned.split()) < 5:
                    continue
                key = norm_text(cleaned)
                with lock_path.open("a+", encoding="utf-8") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    seen = {x.strip() for x in registry_path.read_text(encoding="utf-8").splitlines() if x.strip()}
                    if key in seen:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        continue
                    with registry_path.open("a", encoding="utf-8") as reg:
                        reg.write(key + "\n")
                    fcntl.flock(lock, fcntl.LOCK_UN)
                ordinal = counts[label] + 1
                row = {"sample_id": f"{source}_{LABEL_NAMES[label].lower()}_new_{ordinal:04d}",
                       "source": source, "label": label, "class": LABEL_NAMES[label], "text": cleaned,
                       "generation_seed": sample_seed, "generation_batch": attempt // max(1, args.batch_size),
                       "prompt_version": PROMPT_VERSION, "model_checkpoint": str(model_dir.resolve()),
                       "is_legacy_candidate": False}
                jsonl_append(shard, row)
                counts[label] += 1
                if counts[label] % 25 == 0:
                    print(f"{source}/{LABEL_NAMES[label]}: {counts[label]}/{args.new_per_class}", flush=True)
            if counts[label] != args.new_per_class:
                raise RuntimeError(f"Could not generate {args.new_per_class} unique records for {source}/{LABEL_NAMES[label]}; got {counts[label]}")
    finally:
        del model
    print(json.dumps(dict(counts), indent=2), flush=True)


def compute_roberta(records: list[dict[str, Any]], model_name: str, batch_size: int = 32, device_name: str = "auto") -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval(); outputs = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            enc = tok([x["text"] for x in records[i:i + batch_size]], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs.append(model(**enc).last_hidden_state[:, 0, :].cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs).astype(np.float32)


def compute_minilm(texts: list[str], model_name: str, batch_size: int = 64, device_name: str = "auto") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    device = None if device_name == "auto" else device_name
    model = SentenceTransformer(model_name, local_files_only=True, device=device)
    result = np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    del model
    try:
        import torch
        if device_name.startswith("cuda") or (device_name == "auto" and torch.cuda.is_available()):
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return result


def entropy(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    p = counts[counts > 0] / len(labels)
    return float(-(p * np.log(p)).sum())


def pool_member_ids(membership: pd.DataFrame, source: str, label: int, pool: int) -> set[str]:
    class_name = LABEL_NAMES[int(label)]
    column = f"in_pool{pool}"
    return set(membership[(membership.source == source) & (membership["class"] == class_name) & membership[column]].sample_id)


def lexical_diversity(texts: pd.Series) -> tuple[float, float]:
    """Compute unigram/bigram type ratios once per diagnostic subset."""
    unigram_types: set[str] = set()
    bigram_types: set[tuple[str, str]] = set()
    unigram_total = 0
    bigram_total = 0
    for text in texts.astype(str):
        tokens = text.lower().split()
        unigram_total += len(tokens)
        unigram_types.update(tokens)
        if len(tokens) > 1:
            bigrams = zip(tokens, tokens[1:])
            bigram_types.update(bigrams)
            bigram_total += len(tokens) - 1
    return (len(unigram_types) / max(1, unigram_total), len(bigram_types) / max(1, bigram_total))


def load_expanded(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = frozen_data(); rows = old_candidates(data)
    for source in ("qwen", "llama"):
        rows.extend(jsonl_read(OUT / "generation" / f"{source}_new.jsonl"))
    by_key = collections.defaultdict(list)
    for row in rows: by_key[(row["source"], int(row["label"]))].append(row)
    for key, group in by_key.items():
        if len(group) != 500 or len({x["sample_id"] for x in group}) != 500:
            raise RuntimeError(f"Expanded pool incomplete for {key}: {len(group)}")
    return rows, by_key


def build_analysis(args: argparse.Namespace, include_training: bool = False) -> dict[str, Any]:
    set_output_dir(args.data_dir)
    ensure_disk_budget()
    out = OUT
    rows, groups = load_expanded(args)
    data = frozen_data()
    old_ids = {x["sample_id"] for x in old_candidates(data)}
    real_text = {norm_text(x["text"]) for x in jsonl_read(PROTOCOL / "real_train.jsonl")}
    dev_text = {norm_text(x) for x in pd.read_csv(FROZEN / "dev2000.csv")["text"].tolist()}
    overlaps = {norm_text(x["text"]) for x in rows} & (real_text | dev_text)
    if overlaps: raise RuntimeError(f"Exact leakage found after generation: {len(overlaps)} texts")
    manifest_path = out / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"status": "UNKNOWN_OR_CHANGED"}
    manifest.update({"final_unique_candidate_count": len(rows), "counts_by_source_class": {f"{s}_{LABEL_NAMES[l]}": len(groups[(s, l)]) for s in ("qwen", "llama") for l in LABEL_NAMES},
                     "exact_overlap_real_reference": 0, "exact_overlap_dev2000": 0, "final_holdout_accessed": False,
                     "synthetic_test_used": False, "legacy_ids_preserved": True, "near_duplicates_removed": False,
                     "near_duplicate_policy": "diagnostic only; no semantic near-duplicate filtering"})
    atomic_json_write(manifest_path, manifest)
    model_name = args.roberta_model
    emb = compute_roberta(rows, model_name, args.embedding_batch_size, args.device)
    old_real = data["real_reference"]
    real_emb = compute_roberta(old_real, model_name, args.embedding_batch_size, args.device)
    labels_real = np.asarray([x["label"] for x in old_real])
    prox = []
    for i, row in enumerate(rows):
        ref = real_emb[labels_real == int(row["label"])]
        a = emb[i] / max(np.linalg.norm(emb[i]), 1e-12); b = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        sims = np.sort(a @ b.T)[::-1][:5]
        prox.append(float(sims.mean()))
    for row, score in zip(rows, prox): row["proximity_score"] = score
    membership = []; pool_rows = []
    for source in ("qwen", "llama"):
        for label in LABEL_NAMES:
            old = [x for x in groups[(source, label)] if x["sample_id"] in old_ids]
            new = [x for x in groups[(source, label)] if x["sample_id"] not in old_ids]
            rng = random.Random(EXPANSION_SEED + label * 1009 + (0 if source == "qwen" else 50000)); perm = list(new); rng.shuffle(perm)
            ordered = old + perm
            for rank, row in enumerate(ordered):
                flags = {"in_pool80": rank < 80, "in_pool160": rank < 160, "in_pool250": rank < 250, "in_pool500": rank < 500}
                membership.append({"sample_id": row["sample_id"], "source": source, "class": LABEL_NAMES[label], **flags})
                pool_rows.append(row)
    pd.DataFrame(membership).to_csv(out / "candidate_pool_membership.csv", index=False)
    frame = pd.DataFrame(rows)
    frame["word_count"] = frame.text.map(lambda x: len(str(x).split()))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    frame["encoder_token_count"] = [len(tok(str(x), truncation=True, max_length=512)["input_ids"]) for x in frame.text]
    frame["nearest_synthetic_cosine"] = np.nan
    mini = compute_minilm(frame.text.tolist(), args.minilm_model, args.embedding_batch_size, args.device)
    for (source, label), idxs in frame.groupby(["source", "label"]).groups.items():
        ix = np.asarray(list(idxs)); sims = mini[ix] @ mini[ix].T; np.fill_diagonal(sims, -1)
        frame.loc[ix, "nearest_synthetic_cosine"] = sims.max(axis=1)
    frame["is_legacy_candidate"] = frame.sample_id.isin(old_ids)
    frame.to_csv(out / "expanded_candidate_pool.csv", index=False)
    # Rank and source quality curves.
    rank_rows = []; curve_rows = []
    for (source, label), g in frame.groupby(["source", "label"]):
        s = g.sort_values(["proximity_score", "sample_id"], ascending=[False, True]).reset_index(drop=True)
        for i, value in enumerate(s.proximity_score, 1): rank_rows.append({"rank": i, "source": source, "class": LABEL_NAMES[int(label)], "proximity": value})
        curve_rows.append({"Class": LABEL_NAMES[int(label)], "Source": source, **{f"Top{k}Mean": float(s.proximity_score.head(k).mean()) for k in [20, 40, 60, 80, 100, 160]}})
    pd.DataFrame(rank_rows).to_csv(out / "proximity_by_rank.csv", index=False)
    curve = pd.DataFrame(curve_rows)
    # Per-pool/source/class diagnostics and fixed Pool500 clustering projections.
    mini_cluster = {}; diagnostics = []
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    for (source, label), g in frame.groupby(["source", "label"]):
        idx = np.asarray(g.index); pca = PCA(n_components=0.95, random_state=42).fit(mini[idx]); z = pca.transform(mini[idx]); k = max(2, min(16, round(math.sqrt(len(idx) / 2))))
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(z); mini_cluster[(source, label)] = (pca, km, idx)
        for pool in POOL_SIZES:
            selected_ids = pool_member_ids(pd.DataFrame(membership), source, label, pool)
            subset = g[g.sample_id.isin(selected_ids)]; ix = np.asarray(subset.index); vals = subset.proximity_score.to_numpy(float); wc = subset.word_count.to_numpy(float); cl = km.predict(pca.transform(mini[ix]));
            unigram_ratio, bigram_ratio = lexical_diversity(subset.text)
            diagnostics.append({"PoolSize": pool, "Source": source, "Class": LABEL_NAMES[label], "N": len(subset),
                                "word_mean": wc.mean(), "word_median": np.median(wc), "word_p10": np.percentile(wc, 10), "word_p90": np.percentile(wc, 90),
                                "proximity_mean": vals.mean(), "proximity_median": np.median(vals), "proximity_std": vals.std(ddof=1), "proximity_p10": np.percentile(vals, 10), "proximity_p90": np.percentile(vals, 90), "proximity_max": vals.max(),
                                "pairwise_cosine_mean": float((mini[ix] @ mini[ix].T)[np.triu_indices(len(ix), 1)].mean()) if len(ix) > 1 else 1.0,
                                "nearest_neighbor_cosine_mean": float(subset.nearest_synthetic_cosine.mean()),
                                "unique_unigram_ratio": unigram_ratio,
                                "unique_bigram_ratio": bigram_ratio,
                                "near_dup_gt_099": float((subset.nearest_synthetic_cosine > .99).mean()), "near_dup_gt_0995": float((subset.nearest_synthetic_cosine > .995).mean()), "near_dup_gt_0999": float((subset.nearest_synthetic_cosine > .999).mean()),
                                "cluster_entropy": entropy(cl), "max_cluster_share": float(np.bincount(cl).max() / len(cl)), "occupied_clusters": int(len(set(cl)))})
    diag = pd.DataFrame(diagnostics)
    # Lexical mode counts requested in the prompt.
    terms = {"Llama_SciTech": ["quantum", "quantum computing", "breakthrough", "google", "chip", "ai", "microsoft"], "Llama_Sports": ["lakers", "celtics", "lebron", "nba", "football", "soccer", "tennis"]}
    mode_rows = []
    for key, words in terms.items():
        source, class_name = "llama", key.split("_")[1]; label = next(k for k, v in LABEL_NAMES.items() if v == class_name)
        g = frame[(frame.source == source) & (frame.label == label)]
        for pool in POOL_SIZES:
            ids = pool_member_ids(pd.DataFrame(membership), source, label, pool); subset = g[g.sample_id.isin(ids)]
            for term in words: mode_rows.append({"Focus": key, "PoolSize": pool, "Term": term, "Count": int(subset.text.str.lower().str.contains(re.escape(term), regex=True).sum()), "Proportion": float(subset.text.str.lower().str.contains(re.escape(term), regex=True).mean())})
    # Build selected datasets in memory and save only training outputs.
    selected_sets = {}
    mem = pd.DataFrame(membership)
    for pool in POOL_SIZES:
        selected = []
        for (source, label), g in frame.groupby(["source", "label"]):
            ids = pool_member_ids(mem, source, label, pool)
            selected.extend(g[g.sample_id.isin(ids)].sort_values(["proximity_score", "sample_id"], ascending=[False, True]).head(40).to_dict("records"))
        selected_sets[pool] = selected
    # Persist compact selected IDs only through the run result metadata, not extra CSVs.
    # Diagnostics are embedded in the main workbook/report; avoid emitting a
    # large collection of per-diagnostic CSV files.
    (out / "_selected_sets.json").write_text(json.dumps({str(k): v for k, v in selected_sets.items()}, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    main = build_main(diag, curve, mode_rows, selected_sets, out, include_training)
    write_report(diag, curve, mode_rows, selected_sets, out, include_training)
    return {"selected_sets": selected_sets, "diagnostics": diag, "curve": curve, "mode_rows": mode_rows}


def build_main(diag: pd.DataFrame, curve: pd.DataFrame, mode_rows: list[dict[str, Any]], selected_sets: dict[int, list[dict[str, Any]]], out: Path, include_training: bool) -> pd.DataFrame:
    overview = []
    for pool in POOL_SIZES:
        d = diag[diag.PoolSize == pool]
        selected_scores = [float(x["proximity_score"]) for x in selected_sets[pool]]
        overview.append({"Section": "A_Pool_Overview", "PoolSize": f"Pool{pool}", "Candidates_per_SourceClass": pool, "TotalCandidates": pool * 8, "SelectionRate": 40 / pool,
                         "Qwen_MeanProximity": d[d.Source == "qwen"].proximity_mean.mean(), "Llama_MeanProximity": d[d.Source == "llama"].proximity_mean.mean(),
                         "Qwen_NearDupRate": d[d.Source == "qwen"].near_dup_gt_099.mean(), "Llama_NearDupRate": d[d.Source == "llama"].near_dup_gt_099.mean(),
                         "Qwen_ClusterEntropy": d[d.Source == "qwen"].cluster_entropy.mean(), "Llama_ClusterEntropy": d[d.Source == "llama"].cluster_entropy.mean(),
                         "Selected_MeanProximity": float(np.mean(selected_scores))})
    rows = overview
    if include_training:
        result = pd.read_csv(out / "pool_size_run_results.csv")
        for pool, g in result.groupby("PoolSize"):
            rows.append({"Section": "B_Downstream_Performance", "PoolSize": f"Pool{pool}", "Dev Accuracy mean±std": f"{g.Dev_Accuracy.mean():.6f}±{g.Dev_Accuracy.std(ddof=1):.6f}", "Dev MacroF1 mean±std": f"{g.Dev_Macro_F1.mean():.6f}±{g.Dev_Macro_F1.std(ddof=1):.6f}", "World F1": g.Dev_World_F1.mean(), "Sports F1": g.Dev_Sports_F1.mean(), "Business F1": g.Dev_Business_F1.mean(), "SciTech F1": g.Dev_SciTech_F1.mean(), "Legacy MacroF1 mean": g.Legacy400_Macro_F1.mean()})
        base = result[result.PoolSize == 80].set_index("Model_Seed")
        for a, b in [(160, 80), (250, 80), (500, 80), (250, 160), (500, 250)]:
            xa = result[result.PoolSize == a].set_index("Model_Seed")
            xb = result[result.PoolSize == b].set_index("Model_Seed")
            metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]
            rows.append({"Section": "C_Paired_Delta", "PoolSize": f"{a}-{b}", **{m: (xa[m] - xb[m]).mean() for m in metrics}})
    for _, r in curve.iterrows(): rows.append({"Section": "D_Source_Quality_Curves", "PoolSize": f"{r.Class}-{r.Source}", **{k: r[k] for k in ["Top20Mean", "Top40Mean", "Top60Mean", "Top80Mean", "Top100Mean"]}})
    for key in ["Llama_SciTech", "Llama_Sports"]:
        for pool in POOL_SIZES:
            m = [x for x in mode_rows if x["Focus"] == key and x["PoolSize"] == pool]
            d = diag[(diag.PoolSize == pool) & (diag.Source == "llama") & (diag.Class == key.split("_")[1])].iloc[0]
            row = {"Section": "E_Mode_Collapse", "PoolSize": f"{key}-Pool{pool}", "MaxClusterShare": d.max_cluster_share, "ClusterEntropy": d.cluster_entropy, "NearDupRate": d.near_dup_gt_099}
            for term in ("quantum", "breakthrough", "lakers", "nba"):
                row[f"{term}%"] = next((x["Proportion"] for x in m if x["Term"] == term), np.nan)
            rows.append(row)
    result = pd.DataFrame(rows); result.to_excel(out / "CANDIDATE_POOL_MAIN.xlsx", sheet_name="00_Main", index=False)
    return result


def write_report(diag: pd.DataFrame, curve: pd.DataFrame, modes: list[dict[str, Any]], selected: dict[int, list[dict[str, Any]]], out: Path, include_training: bool) -> None:
    lines = ["Expanded candidate-pool analysis", "", "Offline diagnostics use frozen Real-Anchored Top-5 proximity; final holdout was not accessed."]
    q = diag[diag.Source == "qwen"].groupby("PoolSize").proximity_mean.mean(); l = diag[diag.Source == "llama"].groupby("PoolSize").proximity_mean.mean()
    lines += [f"1. Mean proximity Qwen Pool80->500: {q.loc[80]:.6f} -> {q.loc[500]:.6f}; Llama: {l.loc[80]:.6f} -> {l.loc[500]:.6f}."]
    for key in ("Llama_SciTech", "Llama_Sports"):
        vals = [x for x in modes if x["Focus"] == key and x["Term"] in (["quantum"] if key.endswith("SciTech") else ["nba"])]
        lines.append(f"2. {key} lexical mode proportions: " + ", ".join(f"Pool{x['PoolSize']}={x['Proportion']:.3f}" for x in vals))
    nd = diag.groupby("PoolSize").near_dup_gt_099.mean(); lines.append("3. Near-duplicate >.99 mean by pool: " + ", ".join(f"{p}={nd.loc[p]:.3f}" for p in POOL_SIZES))
    for source in ("qwen", "llama"):
        g = curve[curve.Source == source]; lines.append(f"4. {source} Top40 mean change Pool80->500 by class: " + ", ".join(f"{r.Class}={r.Top40Mean:.6f}" for _, r in g.iterrows()))
    for a, b in [(80, 160), (160, 250), (250, 500)]:
        aa = {x["sample_id"] for x in selected[a]}; bb = {x["sample_id"] for x in selected[b]}
        lines.append(f"4b. Selected TopK overlap Pool{a} vs Pool{b}: count={len(aa & bb)}, Jaccard={len(aa & bb) / max(1, len(aa | bb)):.6f}.")
    if include_training:
        r = pd.read_csv(out / "pool_size_run_results.csv"); base = r[r.PoolSize == 80].set_index("Model_Seed")
        for pool in [160, 250, 500]: lines.append(f"{5 + [160,250,500].index(pool)}. Pool{pool}-Pool80 accuracy mean delta={(r[r.PoolSize == pool].set_index('Model_Seed').Dev_Accuracy - base.Dev_Accuracy).mean():.6f}; Macro-F1 delta={(r[r.PoolSize == pool].set_index('Model_Seed').Dev_Macro_F1 - base.Dev_Macro_F1).mean():.6f}.")
        deltas = [(p, (r[r.PoolSize == p].Dev_Macro_F1.mean() - r[r.PoolSize == 80].Dev_Macro_F1.mean())) for p in [160, 250, 500]]
        label = "strong benefit" if deltas[-1][1] > .01 else "moderate benefit" if deltas[-1][1] > .003 else "saturation around 160" if deltas[0][1] > deltas[1][1] - .001 else "saturation around 250" if deltas[1][1] > deltas[2][1] - .001 else "no meaningful benefit"
        lines.append(f"9. Overall conclusion: {label}.")
    else: lines.append("5-9. Downstream conclusion pending the 9 new BERT runs.")
    (out / "candidate_pool_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_pools(args: argparse.Namespace) -> None:
    set_output_dir(args.data_dir)
    ensure_disk_budget()
    cuda_preflight()
    selected_path = OUT / "_selected_sets.json"
    if selected_path.exists():
        selected_sets = {int(k): v for k, v in json.loads(selected_path.read_text(encoding="utf-8")).items()}
    else:
        selected_sets = build_analysis(args, include_training=False)["selected_sets"]
    out = OUT; dev = pd.read_csv(FROZEN / "dev2000.csv").to_dict("records"); data = frozen_data(); legacy = data["real_test"] if args.evaluate_legacy400 else None
    trainer_args = argparse.Namespace(bert_model=args.bert_model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_length=args.max_length, evaluate_legacy400=args.evaluate_legacy400, evaluate_final_holdout=False)
    rows = []
    # Pool80 is reused only from the frozen E2 results for seeds 42/43/44.
    old = pd.read_csv(FROZEN / "run_results.csv"); old = old[(old.Experiment == "E2") & old.Model_Seed.isin(args.model_seeds)].copy(); old["PoolSize"] = 80; rows.extend(old.to_dict("records"))
    for pool in args.pool_sizes:
        records = selected_sets[pool]
        for seed in args.model_seeds:
            run_name = f"pool{pool}_model{seed}"; result = validation.train_one(records, dev, legacy, None, out, run_name, seed, trainer_args)
            row = validation.metric_row(f"Pool{pool}", "frozen_topk", seed, result, out / "training" / run_name); row["PoolSize"] = pool; rows.append(row)
    part = out / f"pool_size_run_results_{args.shard_name}.csv"; pd.DataFrame(rows).to_csv(part, index=False); print(f"Wrote {part}")


def finalize(args: argparse.Namespace) -> None:
    set_output_dir(args.data_dir)
    ensure_disk_budget()
    parts = sorted(OUT.glob("pool_size_run_results_*.csv"))
    if not parts: raise FileNotFoundError("No training shards found")
    frames = [pd.read_csv(p) for p in parts]; result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["PoolSize", "Model_Seed"], keep="last").sort_values(["PoolSize", "Model_Seed"])
    if len(result) != 12: raise RuntimeError(f"Expected 12 pool-size rows including reused Pool80, found {len(result)}")
    result.to_csv(OUT / "pool_size_run_results.csv", index=False)
    analysis = build_analysis(args, include_training=True); r = result
    rows = []
    metrics = ["Dev_Accuracy", "Dev_Macro_F1", "Dev_World_F1", "Dev_Sports_F1", "Dev_Business_F1", "Dev_SciTech_F1"]
    for a, b in [(160,80),(250,80),(500,80),(250,160),(500,250)]:
        for metric in metrics:
            da = r[r.PoolSize == a].set_index("Model_Seed")[metric] - r[r.PoolSize == b].set_index("Model_Seed")[metric]
            rows.append({"From": a, "To": b, "Metric": metric, "Mean_Delta": da.mean(), "Std_Delta": da.std(ddof=1), "Seed42": da.get(42, np.nan), "Seed43": da.get(43, np.nan), "Seed44": da.get(44, np.nan)})
    pd.DataFrame(rows).to_csv(OUT / "pool_size_paired_deltas.csv", index=False)
    build_main(analysis["diagnostics"], analysis["curve"], analysis["mode_rows"], analysis["selected_sets"], OUT, True)
    write_report(analysis["diagnostics"], analysis["curve"], analysis["mode_rows"], analysis["selected_sets"], OUT, True)
    # Keep only the requested final artifacts; generated shards/registry and
    # selected-set cache are recoverable from the master CSV and membership.
    for path in [OUT / "_selected_sets.json", OUT / "generation" / "qwen_new.jsonl", OUT / "generation" / "llama_new.jsonl", OUT / "generation" / "global_text_registry.txt", OUT / "generation" / "global_text_registry.lock", OUT / "generation" / "generation_manifest.lock"]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    def common(x: argparse.ArgumentParser) -> None:
        x.add_argument("--data-dir", default=str(DEFAULT_OUT)); x.add_argument("--qwen-dir", default=str(ROOT / "cloud/model/Qwen3-4B-Instruct-2507")); x.add_argument("--llama-dir", default=str(ROOT / "cloud/model/Llama-3.2-3B-Instruct")); x.add_argument("--roberta-model", default=str(ROOT / "cloud/model/roberta-base")); x.add_argument("--minilm-model", default=str(ROOT / "cloud/model/all-MiniLM-L6-v2")); x.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"], help="single analysis device; use cuda:0 with CUDA_VISIBLE_DEVICES=0"); x.add_argument("--embedding-batch-size", type=int, default=64); x.add_argument("--bert-model", default=os.environ.get("BERT_MODEL", "bert-base-uncased")); x.add_argument("--evaluate-legacy400", action="store_true"); x.add_argument("--epochs", type=float, default=3); x.add_argument("--batch-size", type=int, default=16); x.add_argument("--learning-rate", type=float, default=2e-5); x.add_argument("--max-length", type=int, default=256)
    g = sub.add_parser("generate"); g.add_argument("--data-dir", default=str(DEFAULT_OUT)); g.add_argument("--source", choices=["qwen","llama"], required=True); g.add_argument("--new-per-class", type=int, default=420); g.add_argument("--max-attempts-per-class", type=int, default=1600); g.add_argument("--max-new-tokens", type=int, default=128); g.add_argument("--batch-size", type=int, default=16); g.add_argument("--four-bit", action="store_true"); g.add_argument("--qwen-dir", default=str(ROOT / "cloud/model/Qwen3-4B-Instruct-2507")); g.add_argument("--llama-dir", default=str(ROOT / "cloud/model/Llama-3.2-3B-Instruct")); g.set_defaults(func=generate_source)
    a = sub.add_parser("analyze"); common(a); a.set_defaults(func=lambda args: build_analysis(args, False))
    t = sub.add_parser("train"); common(t); t.add_argument("--pool-sizes", nargs="+", type=int, required=True); t.add_argument("--model-seeds", nargs="+", type=int, default=MODEL_SEEDS); t.add_argument("--shard-name", required=True); t.set_defaults(func=train_pools)
    f = sub.add_parser("finalize"); common(f); f.add_argument("--model-seeds", nargs="+", type=int, default=MODEL_SEEDS); f.set_defaults(func=finalize)
    args = p.parse_args(); set_output_dir(getattr(args, "data_dir", None)); args.func(args)


if __name__ == "__main__": main()
