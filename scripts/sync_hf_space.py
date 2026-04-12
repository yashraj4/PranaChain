#!/usr/bin/env python3
"""Upload repo root to Hugging Face Space (requires HF_TOKEN in environment)."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = os.getenv("HF_SPACE_REPO", "Yashraj-90/pranachain-openenv")
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN, then run: python scripts/sync_hf_space.py", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=[
            ".git/*",
            "**/__pycache__/*",
            "*.log",
            "*.pyc",
            ".venv/*",
            "venv/*",
        ],
    )
    print(f"Uploaded to https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
