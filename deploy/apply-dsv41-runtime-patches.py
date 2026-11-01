#!/usr/bin/env python3
"""Apply the reviewed durable-cache patch to an exported kvfix4 engine source tree.

The source export must already contain the existing FP8, AOT and kvfix4 patches.
Dry-run validation happens before mutation; repeated application is harmless.
"""

import argparse
import subprocess
from pathlib import Path


def apply(source: Path, patch: Path) -> None:
    command = ["patch", "--batch", "-p1", "-d", str(source)]
    content = patch.read_bytes()
    check = subprocess.run([*command, "--forward", "--dry-run"], input=content, capture_output=True)
    if check.returncode:
        reverse = subprocess.run(
            [*command, "--reverse", "--dry-run"], input=content, capture_output=True
        )
        if reverse.returncode == 0:
            print("Durable-cache patch already applied")
            return
        raise SystemExit("Source differs from the reviewed kvfix4 export; no patch applied")
    subprocess.run([*command, "--forward"], input=content, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    for name in ("0001-durable-kv-checkpoints.patch", "0002-durable-reasoning-memory.patch"):
        apply(args.source.resolve(), Path(__file__).parent / "runtime-patches" / name)


if __name__ == "__main__":
    main()
