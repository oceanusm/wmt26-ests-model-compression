# ESTS GPT-OSS ZHO k26

Primary ESTS English-to-Simplified-Chinese submission for the WMT26 Model
Compression unconstrained track.

- Base model: `openai/gpt-oss-20b`
- Submitted model: `oceanusm/ests-gptoss-zho-k26`
- Language pair: `eng-zho_Hans`
- Structured pruning: average 26 of 32 routed experts removed per layer
- Average retained routed experts: 6 per layer
- Recovery: full-parameter SFT on GPT-5.1 pseudo-parallel proxy data
- Final artifact: MXFP4 MoE weights, approximately 5.15 GiB on disk
- Runtime: vLLM 0.11.1, TP1, 1 GiB fixed KV cache

The runner infers the source category from the line itself because the organizer
interface supplies source lines without the original WMT26 category or
instruction. It uses the exact category instructions used during recovery
training. JSON string leaves are projected to paragraph blocks, translated with
the news instruction, and reconstructed into source-owned JSON. Invalid or
degenerate generations use the same bounded retry, segmentation, placeholder
repair, and source-owned reconstruction policy used in our internal evaluation.

## Setup

```bash
bash setup.sh
```

`setup.sh` creates a local virtual environment and downloads the model into the
shared `MODEL_CACHE`, exposing it at `workdir/model`. Set `MODEL_DIR` to use an
already prepared checkpoint and `MODELZIP_SOURCE` when this directory is run
outside the organizer repository.

## Run

```bash
bash run.sh \
  --lang-pair eng-zho_Hans \
  --batch-size 16 \
  --input input.txt \
  --output output.txt
```

The output contains exactly one physical line per input line. All diagnostics
are written to stderr.
