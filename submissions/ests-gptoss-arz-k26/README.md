# ESTS GPT-OSS ARZ k26

Contrastive ESTS English-to-Egyptian-Arabic submission for the WMT26 Model
Compression unconstrained track.

- Base model: `openai/gpt-oss-20b`
- Submitted model: `oceanusm/ests-gptoss-arz-k26`
- Organizer language pair: `eng-ara_EG`
- Internal target identifier used during recovery: `arz_Arab`
- Structured pruning: average 26 of 32 routed experts removed per layer
- Average retained routed experts: 6 per layer
- Recovery: full-parameter SFT on GPT-5.1 pseudo-parallel proxy data
- Final artifact: MXFP4 MoE weights, approximately 5.15 GiB on disk
- Runtime: vLLM 0.11.1, TP1, 1 GiB fixed KV cache

The runner infers news, social, speech, and JSON inputs from source structure,
uses the exact prompts seen during recovery training, and applies bounded retry,
segmentation, placeholder repair, and source-owned reconstruction.

```bash
bash setup.sh
bash run.sh --lang-pair eng-ara_EG --batch-size 16 --input input.txt --output output.txt
```

Set `MODEL_DIR` to use an existing checkpoint, `MODEL_CACHE` to change the
download cache, and `MODELZIP_SOURCE` when running outside the organizer repo.
The output contains exactly one physical line per input line; diagnostics use
stderr.
