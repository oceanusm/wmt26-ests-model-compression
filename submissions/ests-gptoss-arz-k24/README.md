# baseline

Uncompressed constrained-track baseline for WMT26 Model Compression.

- Base model: `google/gemma-3-12b-it`
- Runtime: PyTorch + Transformers
- Model artifact: uncompressed Gemma 3 model at `workdir/model`, or set `MODEL_DIR` when running.

## Setup

```bash
bash setup.sh
```

`setup.sh` installs this submission's runtime dependencies and the organizer `modelzip` helper package into `./.venv`. When running outside the organizer repository, set `MODELZIP_SOURCE`.

## Compress

```bash
bash compress.sh
```

This baseline is uncompressed, so `compress.sh` is intentionally a no-op.

## Run

```bash
bash run.sh --lang-pair ces-deu --batch-size 1 --input input.txt --output output.txt
```
