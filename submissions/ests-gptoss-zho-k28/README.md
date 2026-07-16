# ESTS GPT-OSS ZHO k28

Contrastive ESTS English-to-Simplified-Chinese submission for the WMT26 Model
Compression unconstrained track.

- Base model: `openai/gpt-oss-20b`
- Submitted model: `oceanusm/ests-gptoss-zho-k28`
- Language pair: `eng-zho_Hans`
- Structured pruning: average 28 of 32 routed experts removed per layer
- Average retained routed experts: 4 per layer
- Recovery: full-parameter SFT on GPT-5.1 pseudo-parallel proxy data
- Final artifact: MXFP4 MoE weights, approximately 4.55 GiB on disk
- Runtime: vLLM 0.11.1, TP1, 1 GiB fixed KV cache

The runner infers news, social, speech, and JSON inputs from source structure,
uses the exact prompts seen during recovery training, and applies bounded retry,
segmentation, placeholder repair, and source-owned reconstruction.

```bash
bash setup.sh
bash run.sh --lang-pair eng-zho_Hans --batch-size 16 --input input.txt --output output.txt
```

Set `MODEL_DIR` to use an existing checkpoint, `MODEL_CACHE` to change the
download cache, and `MODELZIP_SOURCE` when running outside the organizer repo.
The output contains exactly one physical line per input line; diagnostics use
stderr.
