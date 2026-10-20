"""Download models through the hf-mirror.cn-compatible Hugging Face endpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    default_model_root = Path(__file__).resolve().parents[2] / "cloud" / "model"
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("SMAEXP_MODEL_ROOT", str(default_model_root))))
    parser.add_argument("--qwen-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--gpt-id", default="gustavecortal/gpt-neo-2.7B-8bit")
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    args.model_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_qwen:
        print(f"Downloading/resuming {args.qwen_id} ...")
        snapshot_download(
            repo_id=args.qwen_id,
            local_dir=args.model_root / "Qwen3-4B-Instruct-2507",
            resume_download=True,
            max_workers=4,
        )
    print(f"Downloading/resuming {args.gpt_id} ...")
    snapshot_download(
        repo_id=args.gpt_id,
        local_dir=args.model_root / "gpt-neo-2.7B-8bit",
        resume_download=True,
        max_workers=1,
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
        ],
    )
    # The 8-bit repository contains weights/config only; use the official tokenizer.
    snapshot_download(
        repo_id="EleutherAI/gpt-neo-2.7B",
        local_dir=args.model_root / "gpt-neo-2.7B-8bit",
        resume_download=True,
        max_workers=1,
        allow_patterns=["merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
    )


if __name__ == "__main__":
    main()
