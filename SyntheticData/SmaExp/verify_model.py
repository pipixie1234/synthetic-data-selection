"""Verify a sharded Hugging Face model without loading all tensors into RAM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir: Path) -> int:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"FAIL: missing {index_path.name}")
        return 1
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_total = int(index.get("metadata", {}).get("total_size", 0))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    failed = False
    actual_total = 0
    print(f"Model: {model_dir}")
    print(f"Indexed total size: {expected_total:,} bytes")
    for name in shard_names:
        path = model_dir / name
        incomplete = list(model_dir.glob(name + ".incomplete"))
        if not path.exists():
            print(f"FAIL: missing shard {name}")
            if incomplete:
                print(f"      found partial file: {incomplete[0].name} ({incomplete[0].stat().st_size:,} bytes)")
            failed = True
            continue
        size = path.stat().st_size
        actual_total += size
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                tensor_count = len(handle.keys())
            parse_status = f"valid safetensors ({tensor_count} tensors)"
        except Exception as exc:  # pragma: no cover - depends on local files
            parse_status = f"INVALID safetensors: {exc}"
            failed = True
        print(f"{name}: {size:,} bytes; {parse_status}")
    if expected_total and actual_total != expected_total:
        print(f"FAIL: present shard bytes {actual_total:,} != indexed {expected_total:,}")
        failed = True
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / required).exists():
            print(f"FAIL: missing metadata file {required}")
            failed = True
    print("RESULT: INCOMPLETE/INVALID" if failed else "RESULT: COMPLETE")
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    raise SystemExit(verify(parser.parse_args().model_dir))
