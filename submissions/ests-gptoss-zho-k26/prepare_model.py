#!/usr/bin/env python3
"""Download and expose one submission model without copying its weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def resolve_model_source(model_id: str, cache_dir: Path) -> Path:
    explicit = Path(model_id).expanduser()
    if explicit.exists():
        return explicit.resolve()
    if explicit.is_absolute():
        raise FileNotFoundError(f"Model path does not exist: {explicit}")

    cached = cache_dir.expanduser() / model_id
    if (cached / "config.json").is_file():
        return cached.resolve()

    cached.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        local_dir=cached,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if not (cached / "config.json").is_file():
        raise FileNotFoundError(f"Downloaded model is missing config.json: {cached}")
    return cached.resolve()


def expose_model(source: Path, output: Path, *, copy: bool) -> None:
    output = output.expanduser()
    if output.exists() and output.resolve() == source.resolve():
        print(f"Model is already available at {output}")
        return
    if (output / "config.json").is_file():
        print(f"Model is already prepared at {output}")
        return
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite incomplete model directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, output)
    else:
        output.symlink_to(source, target_is_directory=True)
    print(f"Prepared model at {output} from {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser()
    if (output / "config.json").is_file():
        print(f"Model is already prepared at {output}")
        return

    submission = json.loads(args.submission_config.read_text(encoding="utf-8"))
    source = resolve_model_source(str(submission["model_repo"]), args.cache_dir)
    expose_model(source, output, copy=args.copy)


if __name__ == "__main__":
    main()
