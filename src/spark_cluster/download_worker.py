"""Isolated Hugging Face downloader that can be paused or terminated by the controller."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("revision")
    parser.add_argument("target")
    parser.add_argument("workers", type=int)
    args = parser.parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=Path(args.target),
        max_workers=args.workers,
        token=os.environ.get("SPARK_CLUSTER_HF_TOKEN") or None,
    )


if __name__ == "__main__":
    main()
