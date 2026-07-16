#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path


def resolve_model_source(model_id: str, cache_dir: Path) -> Path:
    explicit_path = Path(model_id).expanduser()
    if explicit_path.exists():
        return explicit_path
    if explicit_path.is_absolute():
        raise FileNotFoundError(f"Model path does not exist: {explicit_path}")

    cached_model_dir = cache_dir.expanduser() / model_id
    if (cached_model_dir / "config.json").exists():
        return cached_model_dir

    cached_model_dir.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, local_dir=cached_model_dir)
    (cached_model_dir / "._DOWNLOAD_OK").touch()
    return cached_model_dir


def link_model(source: Path, output_dir: Path, copy: bool = False):
    output_dir = output_dir.expanduser()
    if (output_dir / "config.json").exists():
        print(f"Model already prepared at {output_dir}")
        return
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite incomplete model directory: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, output_dir)
    else:
        try:
            output_dir.symlink_to(source.resolve(), target_is_directory=True)
        except OSError:
            shutil.copytree(source, output_dir)
    print(f"Prepared model at {output_dir} from {source}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the uncompressed Gemma baseline model")
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--cache-dir", type=Path, default="/mnt/tg/data/projects/wmt26/model-compression/models")
    parser.add_argument("--output", type=Path, default="workdir/model")
    parser.add_argument("--copy", action="store_true", help="Copy model files instead of symlinking")
    args = parser.parse_args()

    source = resolve_model_source(args.model_id, args.cache_dir)
    link_model(source, args.output, copy=args.copy)


if __name__ == "__main__":
    main()
