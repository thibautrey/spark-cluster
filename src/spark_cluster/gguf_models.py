"""Select one GGUF weight set without mistaking auxiliary files for a model."""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


@dataclass(frozen=True)
class GGUFSelection:
    model_file: str
    files: tuple[str, ...]
    weight_bytes: int
    shared_shard_source: str | None = None
    shared_shard_name: str | None = None


def smallest_gguf(files: tuple[tuple[str, int], ...]) -> GGUFSelection:
    variants: dict[str, list[tuple[str, int]]] = {}
    shared = []
    projectors = []
    for name, size in files:
        basename = PurePosixPath(name).name.lower()
        if not basename.endswith(".gguf"):
            continue
        if basename.startswith("mmproj-"):
            projectors.append(name)
        elif "engram" in basename:
            shared.append((name, size))
        else:
            match = SHARD.fullmatch(name)
            variants.setdefault(match[1] if match else name, []).append((name, size))
    if not variants:
        raise ValueError("No GGUF model weights found")
    key = min(variants, key=lambda name: (sum(size for _, size in variants[name]), name))
    selected = sorted(variants[key])
    first = selected[0][0]
    alias_source = alias_name = None
    shard = SHARD.fullmatch(first)
    if shard:
        total = int(shard[3])
        present = {int(SHARD.fullmatch(name)[2]) for name, _ in selected}
        missing = set(range(1, total + 1)) - present
        if missing == {total} and len(shared) == 1:
            alias_source = shared[0][0]
            alias_name = f"{shard[1]}-{total:05d}-of-{total:05d}.gguf"
        elif missing:
            raise ValueError("Selected GGUF variant is missing model shards")
    return GGUFSelection(
        model_file=first,
        files=tuple(name for name, _ in selected + shared) + tuple(projectors),
        weight_bytes=sum(size for _, size in selected + shared),
        shared_shard_source=alias_source,
        shared_shard_name=alias_name,
    )
