#!/usr/bin/env python3
"""Line-oriented WMT26 inference for physically pruned GPT-OSS specialists."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

from wmt26_json_adapter import (
    WMT26JSONError,
    parse_wmt26_json_source,
    project_json_to_html_robust,
    reconstruct_json_from_html_robust,
    reconstruct_json_from_robust_leaf_values,
    split_json_leaf_around_placeholders,
    validate_robust_json_leaf_translation,
)
from wmt26_robust_inference import (
    WMT26RobustError,
    build_non_json_segmentation_plan,
    extract_exact_paragraph_blocks,
    generation_failure_reasons,
    raw_fallback_rank,
    reconstruct_non_json_segments,
    split_long_text,
    validate_non_json_document,
    validate_segment_output,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "submission.json"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "workdir" / "model"
MAX_DECLARED_BATCH_SIZE = 16
DEFAULT_KV_CACHE_BYTES = 1024**3
DEFAULT_MAX_MODEL_LEN = 12_288
DEFAULT_MAX_NEW_TOKENS = 8_192
DEFAULT_SEGMENT_MAX_NEW_TOKENS = 2_048
DEFAULT_TEMPERATURE = 0.5
WHOLE_DOCUMENT_SEEDS = tuple(range(10))
SEGMENT_SEEDS = tuple(range(5))
PLACEHOLDER_MODE = "raw-repair"
ORIGINAL_NUM_EXPERTS = 32

P_BLOCK_RE = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
ANY_TAG_RE = re.compile(r"<[^>]+>")
SOCIAL_SIGNAL_RE = re.compile(
    r"(?:^|\s)@[A-Za-z0-9_.-]+|#[\w-]+|https?://|www\.|"
    r"\b(?:lol|lmao|omg|wtf)\b|\$partner|\b\d+/x\b|"
    r"[\U0001F300-\U0001FAFF]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubmissionConfig:
    submission_id: str
    model_repo: str
    lang_pair: str
    internal_target_lang: str
    target_display_name: str
    expected_average_drop: int


@dataclass
class PreparedRecord:
    index: int
    source_text: str
    category: str
    model_category: str
    model_source_text: str
    instruction: str
    prompt: str
    json_projection: dict[str, Any] | None = None


@dataclass
class GenerationResult:
    model_hyp_text: str
    raw_comp_text: str
    prompt_token_count: int
    completion_token_count: int
    finish_reason: str | None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_submission_config(path: Path = DEFAULT_CONFIG) -> SubmissionConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "submission_id",
        "model_repo",
        "lang_pair",
        "internal_target_lang",
        "target_display_name",
        "expected_average_drop",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Submission config is missing fields: {missing}")
    return SubmissionConfig(
        submission_id=str(value["submission_id"]),
        model_repo=str(value["model_repo"]),
        lang_pair=str(value["lang_pair"]),
        internal_target_lang=str(value["internal_target_lang"]),
        target_display_name=str(value["target_display_name"]),
        expected_average_drop=int(value["expected_average_drop"]),
    )


def instruction_templates(config: SubmissionConfig) -> dict[str, str]:
    language = config.target_display_name
    code = config.internal_target_lang
    prefix = (
        f"You are a professional {language} translator, tasked with providing "
        f"translations suitable for use in {language} ({code}). Your goal is to "
        "accurately convey the meaning and nuances of the original text while "
        f"adhering to {language} grammar and vocabulary and ensuring that the "
        "translation is natural. "
    )
    speech = prefix + (
        "The original text is automatically transcribed from spoken language and can "
        "contain errors. Maintain the flow and colloquial style of the speaker in the "
        "translation. Do not include non-linguistic sounds (e.g. laughter, groans, "
        "hesitation sounds, etc.), but do include interjections. If a word is "
        "interrupted, either guess the full word if possible or otherwise omit it. "
        "Keep foreign words as they are when translating. Produce only the "
        f"{language} translation, without any additional explanations or commentary. "
        "Output the text such that each sentence is on a separate line. Please "
        f"translate the following text into {language} ({code}):"
    )
    social = prefix + (
        "The original text is user-generated content from a social media platform. "
        "Do not reproduce spelling mistakes. Reproduce marks of expressiveness that "
        "communicate meaningful intent (e.g. enthusiasm through capitalisation or "
        f"elongation) in a way that is natural in {language}. Copy URLs and user "
        "handles directly rather than translating them. However, translate hashtags "
        "as appropriate for the translation to be natural for social media text. "
        "Follow the punctuation of the source text as best as possible. Additional "
        "punctuation should be added only if not doing so would seriously alter the "
        "comprehension of the text. Translate text in an informal style, like close "
        "friends talking, even if it changes the original tone. Produce only the "
        f"{language} translation, without any additional explanations or commentary. "
        "Maintain the HTML formatting of the original source text. Please translate "
        f"the following text into {language} ({code}):"
    )
    news = prefix + (
        "The original text is a news article. Ensure the translation is formal and "
        "consistent with journalistic standards. Produce only the "
        f"{language} translation, without any additional explanations or commentary. "
        "Maintain the HTML formatting of the original source text. Please translate "
        f"the following text into {language} ({code}):"
    )
    return {"speech": speech, "social": social, "news": news}


def build_prompt(instruction: str, source_text: str) -> str:
    return f"{instruction.strip()}\n\n{source_text}"


def _plain_paragraph_text(value: str) -> str:
    return html.unescape(ANY_TAG_RE.sub("", value)).strip()


def _looks_like_news_headline(value: str) -> bool:
    text = _plain_paragraph_text(value)
    if not text or len(text) > 180:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not words:
        return False
    significant = [word for word in words if len(word) > 2]
    title_ratio = (
        sum(word[0].isupper() for word in significant) / len(significant)
        if significant
        else 0.0
    )
    return title_ratio >= 0.55 or (
        len(text) <= 120 and not re.search(r"[.!?]$", text)
    )


def infer_category(source_text: str) -> str:
    stripped = str(source_text).strip()
    if stripped.startswith(("{", "[", "```")):
        try:
            parse_wmt26_json_source(stripped)
            return "json"
        except WMT26JSONError:
            pass

    blocks = P_BLOCK_RE.findall(stripped)
    if blocks or re.search(r"<\s*p\b", stripped, re.IGNORECASE):
        if BR_RE.search(stripped) or SOCIAL_SIGNAL_RE.search(stripped):
            return "social"
        if blocks and len(blocks) <= 5 and not _looks_like_news_headline(blocks[0]):
            return "social"
        return "news"
    return "speech"


def prepare_record(
    index: int,
    source_text: str,
    config: SubmissionConfig,
    templates: Mapping[str, str],
) -> PreparedRecord:
    category = infer_category(source_text)
    model_category = category
    model_source = source_text
    projection = None
    if category == "json":
        model_category = "news"
        model_source, _, projection = project_json_to_html_robust(
            source_text,
            templates["news"],
            placeholder_mode=PLACEHOLDER_MODE,
        )
    instruction = templates[model_category]
    return PreparedRecord(
        index=index,
        source_text=source_text,
        category=category,
        model_category=model_category,
        model_source_text=model_source,
        instruction=instruction,
        prompt=build_prompt(instruction, model_source),
        json_projection=projection,
    )


def validate_model_checkpoint(model_dir: Path, config: SubmissionConfig) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model config is missing at {config_path}. Run setup.sh or set MODEL_DIR."
        )
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_config.get("model_type") != "pruned_gpt_oss":
        raise ValueError(
            f"Expected model_type='pruned_gpt_oss', got {model_config.get('model_type')!r}."
        )
    counts = model_config.get("num_local_experts_per_layer")
    num_layers = int(model_config.get("num_hidden_layers", 0))
    if not isinstance(counts, list) or len(counts) != num_layers or not counts:
        raise ValueError("Invalid or missing num_local_experts_per_layer in config.json.")
    counts = [int(value) for value in counts]
    active = int(model_config.get("num_experts_per_tok", 4))
    if min(counts) < active:
        raise ValueError("A layer retains fewer experts than num_experts_per_tok.")
    average_drop = sum(ORIGINAL_NUM_EXPERTS - value for value in counts) / len(counts)
    if abs(average_drop - config.expected_average_drop) > 1e-9:
        raise ValueError(
            f"Checkpoint average drop is {average_drop}, expected "
            f"{config.expected_average_drop} for {config.submission_id}."
        )
    quantization = model_config.get("quantization_config") or {}
    return {
        "layers": num_layers,
        "average_keep": ORIGINAL_NUM_EXPERTS - average_drop,
        "average_drop": average_drop,
        "min_keep": min(counts),
        "max_keep": max(counts),
        "format": "mxfp4" if quantization.get("quant_method") == "mxfp4" else "bf16",
    }


def patch_tokenizers_backend_for_vllm() -> type:
    """Let Transformers 4.x load tokenizer configs written by Transformers 5.x."""
    import transformers
    from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
    except (ImportError, ModuleNotFoundError):
        # Transformers 5 writes this class name into tokenizer_config.json, while
        # vLLM 0.11.1 requires Transformers 4.x. Both classes load tokenizer.json.
        TokenizersBackend = PreTrainedTokenizerFast

    # AutoTokenizer's 4.x class-name lookup checks the top-level module last.
    setattr(transformers, "TokenizersBackend", TokenizersBackend)

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        def all_special_tokens_extended(self):
            tokens = []
            for attr in getattr(self, "SPECIAL_TOKENS_ATTRIBUTES", []):
                value = getattr(self, attr, None)
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    tokens.extend(value)
                else:
                    tokens.append(value)

            deduped = []
            for token in tokens:
                if token not in deduped:
                    deduped.append(token)
            return deduped

        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            all_special_tokens_extended
        )

    if not hasattr(TokenizersBackend, "all_special_tokens_extended"):
        TokenizersBackend.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    return TokenizersBackend


class GPTOSSGenerator:
    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int,
        temperature: float,
        max_model_len: int,
        kv_cache_memory_bytes: int,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import AutoConfig
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

        patch_tokenizers_backend_for_vllm()

        from vllm import LLM
        from vllm.model_executor.models import ModelRegistry

        class PrunedGptOssVLLMConfig(GptOssConfig):
            model_type = "pruned_gpt_oss"

        AutoConfig.register(
            "pruned_gpt_oss",
            PrunedGptOssVLLMConfig,
            exist_ok=True,
        )
        ModelRegistry.register_model(
            "PrunedGptOssForCausalLM",
            "pruned_gptoss:GptOssForCausalLM",
        )

        self.temperature = float(temperature)
        self.batch_size = int(batch_size)
        self._encoding = None
        self.llm = LLM(
            model=str(model_dir),
            tensor_parallel_size=1,
            gpu_memory_utilization=float(
                os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
            ),
            kv_cache_memory_bytes=int(kv_cache_memory_bytes),
            max_model_len=int(max_model_len),
            max_num_seqs=int(batch_size),
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_lora=False,
            swap_space=float(os.environ.get("VLLM_SWAP_SPACE", "0")),
            hf_overrides={"architectures": ["PrunedGptOssForCausalLM"]},
        )

    @property
    def encoding(self):
        if self._encoding is None:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding

            self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        return self._encoding

    def render_prompt(self, prompt: str) -> list[int]:
        from openai_harmony import (
            Conversation,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
        )

        system_content = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, system_content),
                Message.from_role_and_content(Role.USER, prompt),
            ]
        )
        return self.encoding.render_conversation_for_completion(
            conversation,
            Role.ASSISTANT,
        )

    @staticmethod
    def _looks_like_harmony_scaffold(raw_text: str) -> bool:
        lowered = str(raw_text or "").strip().lower()
        markers = (
            "assistantfinal",
            "assistantanalysis",
            "assistantcommentary",
            "<|channel|>",
            "<|start|>",
            "<|message|>",
        )
        return any(marker in lowered for marker in markers) or lowered.startswith(
            ("analysis", "final", "commentary")
        )

    def parse_completion(
        self,
        token_ids: list[int],
        raw_text: str,
        *,
        finish_reason: str | None,
        max_tokens: int,
    ) -> str:
        from openai_harmony import Role

        try:
            messages = self.encoding.parse_messages_from_completion_tokens(
                token_ids,
                Role.ASSISTANT,
            )
            for message in messages:
                value = message.to_dict()
                if value.get("channel") != "final":
                    continue
                content = value.get("content", [])
                if len(content) == 1 and "text" in content[0]:
                    return str(content[0]["text"])
                return "ERROR: CONTENT HAS MULTIPLE ENTRIES"
        except Exception:
            pass

        raw_text = str(raw_text or "").strip()
        hit_length = finish_reason == "length" or len(token_ids) >= max_tokens
        if raw_text and not hit_length and not self._looks_like_harmony_scaffold(raw_text):
            return raw_text
        if hit_length:
            return "ERROR: HIT MAX TOKENS"
        return "ERROR: NO MESSAGE WITH CHANNEL 'FINAL'"

    def generate(
        self,
        prompts: list[str],
        *,
        seed: int,
        max_tokens: int,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        if not prompts:
            return []
        sampling = SamplingParams(
            temperature=self.temperature,
            max_tokens=int(max_tokens),
            seed=int(seed),
            stop_token_ids=self.encoding.stop_tokens_for_assistant_actions(),
        )
        rendered = [
            {"prompt_token_ids": self.render_prompt(prompt)}
            for prompt in prompts
        ]
        outputs = self.llm.generate(rendered, sampling_params=sampling)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            raw_text = str(completion.text or "")
            finish_reason = getattr(completion, "finish_reason", None)
            results.append(
                GenerationResult(
                    model_hyp_text=self.parse_completion(
                        token_ids,
                        raw_text,
                        finish_reason=finish_reason,
                        max_tokens=max_tokens,
                    ),
                    raw_comp_text=raw_text,
                    prompt_token_count=len(output.prompt_token_ids),
                    completion_token_count=len(token_ids),
                    finish_reason=finish_reason,
                )
            )
        if len(results) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(results)} outputs for {len(prompts)} prompts."
            )
        return results


def batched_generate(
    generator: GPTOSSGenerator,
    items: list[dict[str, Any]],
    *,
    seed: int,
    max_tokens: int,
    batch_size: int,
) -> list[tuple[dict[str, Any], GenerationResult]]:
    results: list[tuple[dict[str, Any], GenerationResult]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        generated = generator.generate(
            [str(item["prompt"]) for item in batch],
            seed=seed,
            max_tokens=max_tokens,
        )
        if len(generated) != len(batch):
            raise RuntimeError(
                f"Generator returned {len(generated)} results for {len(batch)} items."
            )
        results.extend(zip(batch, generated))
    return results


def base_attempt(
    result: GenerationResult,
    *,
    seed: int,
    stage: str,
    max_tokens: int,
) -> dict[str, Any]:
    reasons, repetition = generation_failure_reasons(
        result.model_hyp_text,
        completion_token_count=result.completion_token_count,
        max_tokens=max_tokens,
    )
    if result.finish_reason == "length" and "hit_max_tokens" not in reasons:
        reasons.append("hit_max_tokens")
    return {
        "stage": stage,
        "seed": int(seed),
        "model_hyp_text": result.model_hyp_text,
        "raw_comp_text": result.raw_comp_text,
        "prompt_token_count": result.prompt_token_count,
        "completion_token_count": result.completion_token_count,
        "finish_reason": result.finish_reason,
        "failure_reasons": reasons,
        "repetition": repetition,
        "validation": {},
    }


def validate_whole_attempt(
    record: PreparedRecord,
    attempt: dict[str, Any],
) -> str | None:
    if attempt["failure_reasons"]:
        return None
    output = str(attempt["model_hyp_text"])
    if record.category == "json":
        try:
            _, blocks, html_validation = extract_exact_paragraph_blocks(
                record.model_source_text,
                output,
                allow_anchor=True,
                allow_single_raw=True,
            )
            final_text, json_validation = reconstruct_json_from_html_robust(
                record.source_text,
                blocks,
                projection_meta=record.json_projection or {},
            )
            attempt["validation"] = {
                "html": html_validation,
                "json": json_validation,
            }
            return final_text
        except (WMT26JSONError, WMT26RobustError) as exc:
            attempt["failure_reasons"].append("json_reconstruction_error")
            attempt["validation_error"] = str(exc)
            return None
    try:
        final_text, validation = validate_non_json_document(
            record.source_text,
            output,
            category=record.category,
        )
        attempt["validation"] = validation
        return final_text
    except WMT26RobustError as exc:
        attempt["failure_reasons"].append("html_topology_error")
        attempt["validation_error"] = str(exc)
        return None


def run_whole_document_attempts(
    generator: GPTOSSGenerator,
    records: list[PreparedRecord],
    finals: dict[int, tuple[str, str]],
    attempts: dict[int, list[dict[str, Any]]],
    *,
    batch_size: int,
    max_tokens: int,
    diagnostics: Counter[str],
) -> None:
    for seed in WHOLE_DOCUMENT_SEEDS:
        pending = [record for record in records if record.index not in finals]
        if not pending:
            return
        items = [
            {"record": record, "prompt": record.prompt}
            for record in pending
        ]
        for item, result in batched_generate(
            generator,
            items,
            seed=seed,
            max_tokens=max_tokens,
            batch_size=batch_size,
        ):
            record = item["record"]
            attempt = base_attempt(
                result,
                seed=seed,
                stage="whole_document",
                max_tokens=max_tokens,
            )
            final_text = validate_whole_attempt(record, attempt)
            attempts[record.index].append(attempt)
            if final_text is not None:
                status = "whole_document_success"
                finals[record.index] = (final_text, status)
                diagnostics[status] += 1
        log(f"whole-document seed={seed}: attempted={len(pending)} unresolved={sum(r.index not in finals for r in records)}")


def translate_segment_items(
    generator: GPTOSSGenerator,
    items: list[dict[str, Any]],
    *,
    batch_size: int,
    max_tokens: int,
    diagnostics: Counter[str],
    stage: str,
) -> dict[tuple[int, str], str]:
    successful: dict[tuple[int, str], str] = {}
    for seed in SEGMENT_SEEDS:
        pending = [
            item
            for item in items
            if (item["record"].index, str(item["segment"]["segment_id"]))
            not in successful
        ]
        if not pending:
            break
        for item, result in batched_generate(
            generator,
            pending,
            seed=seed,
            max_tokens=max_tokens,
            batch_size=batch_size,
        ):
            record = item["record"]
            segment = item["segment"]
            segment_id = str(segment["segment_id"])
            attempt = base_attempt(
                result,
                seed=seed,
                stage=stage,
                max_tokens=max_tokens,
            )
            if attempt["failure_reasons"]:
                diagnostics[f"{stage}_generation_failure"] += 1
                continue
            try:
                translated, _ = validate_segment_output(
                    segment,
                    result.model_hyp_text,
                )
                if not translated.strip():
                    raise WMT26RobustError("Segment output is empty after validation.")
                successful[(record.index, segment_id)] = translated
            except WMT26RobustError:
                diagnostics[f"{stage}_structure_failure"] += 1
        log(f"{stage} seed={seed}: attempted={len(pending)} accepted={len(successful)}/{len(items)}")
    return successful


def run_non_json_segmentation(
    generator: GPTOSSGenerator,
    records: list[PreparedRecord],
    finals: dict[int, tuple[str, str]],
    *,
    batch_size: int,
    max_tokens: int,
    diagnostics: Counter[str],
    target_lang: str,
) -> None:
    plans: dict[int, dict[str, Any]] = {}
    unresolved = [
        record
        for record in records
        if record.category != "json" and record.index not in finals
    ]
    for record in unresolved:
        try:
            plans[record.index] = build_non_json_segmentation_plan(
                record.source_text,
                category=record.category,
            )
        except WMT26RobustError:
            diagnostics["segmentation_plan_failure"] += 1

    coarse_items = []
    for record in unresolved:
        plan = plans.get(record.index)
        if plan is None:
            continue
        for container in plan["containers"]:
            segment = container.get("coarse")
            if segment:
                coarse_items.append(
                    {
                        "record": record,
                        "segment": segment,
                        "prompt": build_prompt(record.instruction, segment["model_source_text"]),
                    }
                )
    coarse = translate_segment_items(
        generator,
        coarse_items,
        batch_size=batch_size,
        max_tokens=max_tokens,
        diagnostics=diagnostics,
        stage="segment_coarse",
    )

    fine_items = []
    for record in unresolved:
        plan = plans.get(record.index)
        if plan is None:
            continue
        for container in plan["containers"]:
            coarse_segment = container.get("coarse")
            if coarse_segment and (
                record.index,
                str(coarse_segment["segment_id"]),
            ) in coarse:
                continue
            segments = (
                [span for line in container["fine"] for span in line["spans"]]
                if plan["kind"] == "html"
                else list(container["fine"])
            )
            for segment in segments:
                fine_items.append(
                    {
                        "record": record,
                        "segment": segment,
                        "prompt": build_prompt(record.instruction, segment["model_source_text"]),
                    }
                )
    fine = translate_segment_items(
        generator,
        fine_items,
        batch_size=batch_size,
        max_tokens=max_tokens,
        diagnostics=diagnostics,
        stage="segment_fine",
    )

    all_translations = {**coarse, **fine}
    for record in unresolved:
        plan = plans.get(record.index)
        if plan is None:
            continue
        translations = {
            segment_id: value
            for (index, segment_id), value in all_translations.items()
            if index == record.index
        }
        try:
            reconstructed = reconstruct_non_json_segments(
                plan,
                translations,
                target_lang=target_lang,
            )
            final_text, _ = validate_non_json_document(
                record.source_text,
                reconstructed,
                category=record.category,
            )
            finals[record.index] = (final_text, "segmented_success")
            diagnostics["segmented_success"] += 1
        except WMT26RobustError:
            diagnostics["segmented_reconstruction_failure"] += 1


def translate_json_leaves(
    generator: GPTOSSGenerator,
    records: list[PreparedRecord],
    finals: dict[int, tuple[str, str]],
    *,
    batch_size: int,
    max_tokens: int,
    diagnostics: Counter[str],
) -> dict[tuple[int, int], str]:
    items = []
    for record in records:
        if record.category != "json" or record.index in finals:
            continue
        for leaf in (record.json_projection or {}).get("leaves", []):
            items.append(
                {
                    "record": record,
                    "leaf": leaf,
                    "leaf_index": int(leaf["leaf_index"]),
                    "prompt": build_prompt(record.instruction, leaf["projected_html"]),
                }
            )
    successful: dict[tuple[int, int], str] = {}
    for seed in SEGMENT_SEEDS:
        pending = [
            item
            for item in items
            if (item["record"].index, item["leaf_index"]) not in successful
        ]
        if not pending:
            break
        for item, result in batched_generate(
            generator,
            pending,
            seed=seed,
            max_tokens=max_tokens,
            batch_size=batch_size,
        ):
            record = item["record"]
            leaf = item["leaf"]
            key = (record.index, int(item["leaf_index"]))
            attempt = base_attempt(
                result,
                seed=seed,
                stage="json_leaf",
                max_tokens=max_tokens,
            )
            if attempt["failure_reasons"]:
                diagnostics["json_leaf_generation_failure"] += 1
                continue
            try:
                _, blocks, _ = extract_exact_paragraph_blocks(
                    leaf["projected_html"],
                    result.model_hyp_text,
                    allow_anchor=True,
                    allow_single_raw=True,
                )
                if len(blocks) != 1:
                    raise WMT26RobustError(
                        f"Expected one translated JSON leaf, got {len(blocks)}."
                    )
                value = BR_RE.sub("\n", blocks[0])
                value = html.unescape(value).strip()
                value, _ = validate_robust_json_leaf_translation(
                    leaf,
                    value,
                    placeholder_mode=PLACEHOLDER_MODE,
                )
                successful[key] = value
            except (WMT26JSONError, WMT26RobustError):
                diagnostics["json_leaf_reconstruction_failure"] += 1
        log(f"json-leaf seed={seed}: attempted={len(pending)} accepted={len(successful)}/{len(items)}")
    return successful


def run_json_fallback(
    generator: GPTOSSGenerator,
    records: list[PreparedRecord],
    finals: dict[int, tuple[str, str]],
    *,
    batch_size: int,
    max_tokens: int,
    diagnostics: Counter[str],
    target_lang: str,
) -> None:
    leaf_success = translate_json_leaves(
        generator,
        records,
        finals,
        batch_size=batch_size,
        max_tokens=max_tokens,
        diagnostics=diagnostics,
    )
    span_items = []
    for record in records:
        if record.category != "json" or record.index in finals:
            continue
        for leaf in (record.json_projection or {}).get("leaves", []):
            leaf_index = int(leaf["leaf_index"])
            if (record.index, leaf_index) in leaf_success:
                continue
            for part_index, part in enumerate(
                split_json_leaf_around_placeholders(str(leaf["source_value"]))
            ):
                if part["kind"] != "text" or not part["value"].strip():
                    continue
                for span_index, span in enumerate(split_long_text(part["value"].strip())):
                    segment = {
                        "segment_id": f"json{leaf_index}.part{part_index}.span{span_index}",
                        "model_source_text": f"<p>{html.escape(span, quote=False)}</p>",
                        "source_text": span,
                        "wrapper": "paragraph_plain",
                    }
                    span_items.append(
                        {
                            "record": record,
                            "segment": segment,
                            "prompt": build_prompt(
                                record.instruction,
                                segment["model_source_text"],
                            ),
                        }
                    )
    span_success = translate_segment_items(
        generator,
        span_items,
        batch_size=batch_size,
        max_tokens=max_tokens,
        diagnostics=diagnostics,
        stage="json_span",
    )

    joiner = "" if target_lang == "zho_Hans" else " "
    for record in records:
        if record.category != "json" or record.index in finals:
            continue
        values: dict[int, str] = {}
        source_fallback_used = False
        for leaf in (record.json_projection or {}).get("leaves", []):
            leaf_index = int(leaf["leaf_index"])
            leaf_key = (record.index, leaf_index)
            if leaf_key in leaf_success:
                values[leaf_index] = leaf_success[leaf_key]
                continue
            rendered_parts = []
            for part_index, part in enumerate(
                split_json_leaf_around_placeholders(str(leaf["source_value"]))
            ):
                if part["kind"] == "placeholder" or not part["value"].strip():
                    rendered_parts.append(part["value"])
                    continue
                leading = part["value"][: len(part["value"]) - len(part["value"].lstrip())]
                trailing = part["value"][len(part["value"].rstrip()) :]
                translated_spans = []
                for span_index, source_span in enumerate(
                    split_long_text(part["value"].strip())
                ):
                    segment_id = f"json{leaf_index}.part{part_index}.span{span_index}"
                    translated = span_success.get((record.index, segment_id))
                    if translated is None:
                        translated = source_span
                        source_fallback_used = True
                    translated_spans.append(translated)
                rendered_parts.append(
                    leading + joiner.join(translated_spans) + trailing
                )
            value = "".join(rendered_parts)
            value, _ = validate_robust_json_leaf_translation(
                {**leaf, "placeholder_records": []},
                value,
                placeholder_mode=PLACEHOLDER_MODE,
            )
            values[leaf_index] = value
        final_text, _ = reconstruct_json_from_robust_leaf_values(
            record.source_text,
            values,
        )
        status = "source_text_fallback" if source_fallback_used else "json_leaf_success"
        finals[record.index] = (final_text, status)
        diagnostics[status] += 1


def apply_raw_fallbacks(
    records: list[PreparedRecord],
    finals: dict[int, tuple[str, str]],
    attempts: Mapping[int, list[dict[str, Any]]],
    diagnostics: Counter[str],
) -> None:
    for record in records:
        if record.index in finals:
            continue
        candidates = [
            attempt
            for attempt in attempts[record.index]
            if str(attempt.get("model_hyp_text") or "").strip()
        ]
        if record.category != "json" and candidates:
            selected = max(candidates, key=raw_fallback_rank)
            finals[record.index] = (
                str(selected["model_hyp_text"]).strip(),
                "raw_completion_fallback",
            )
            diagnostics["raw_completion_fallback"] += 1
        else:
            finals[record.index] = (record.source_text, "source_text_fallback")
            diagnostics["source_text_fallback"] += 1


def compact_one_line(text: str, *, category: str) -> str:
    text = str(text or "")
    if category == "json":
        try:
            value = json.loads(text)
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"Final JSON output is invalid: {exc}") from exc
    return re.sub(r"[\r\n]+", " ", text).strip()


def translate_lines(
    generator: GPTOSSGenerator,
    source_lines: list[str],
    config: SubmissionConfig,
    *,
    batch_size: int,
    max_new_tokens: int,
    segment_max_new_tokens: int,
) -> tuple[list[str], dict[str, Any]]:
    templates = instruction_templates(config)
    records = [
        prepare_record(index, source, config, templates)
        for index, source in enumerate(source_lines)
        if source.strip()
    ]
    finals: dict[int, tuple[str, str]] = {}
    attempts: dict[int, list[dict[str, Any]]] = {
        record.index: [] for record in records
    }
    diagnostics: Counter[str] = Counter()
    diagnostics.update(
        {f"category_{key}": value for key, value in Counter(r.category for r in records).items()}
    )

    run_whole_document_attempts(
        generator,
        records,
        finals,
        attempts,
        batch_size=batch_size,
        max_tokens=max_new_tokens,
        diagnostics=diagnostics,
    )
    run_non_json_segmentation(
        generator,
        records,
        finals,
        batch_size=batch_size,
        max_tokens=segment_max_new_tokens,
        diagnostics=diagnostics,
        target_lang=config.internal_target_lang,
    )
    run_json_fallback(
        generator,
        records,
        finals,
        batch_size=batch_size,
        max_tokens=segment_max_new_tokens,
        diagnostics=diagnostics,
        target_lang=config.internal_target_lang,
    )
    apply_raw_fallbacks(records, finals, attempts, diagnostics)

    outputs = [""] * len(source_lines)
    by_index = {record.index: record for record in records}
    for index, (text, status) in finals.items():
        outputs[index] = compact_one_line(text, category=by_index[index].category)
        diagnostics[f"final_{status}"] += 1
    if len(outputs) != len(source_lines):
        raise RuntimeError("Output line count does not match input line count.")
    return outputs, dict(sorted(diagnostics.items()))


def read_input_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Input file is empty: {path}")
    return lines


def atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for line in lines:
                if "\n" in line or "\r" in line:
                    raise ValueError("Output line still contains a physical newline.")
                handle.write(line)
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pos_lang_pair", nargs="?")
    parser.add_argument("pos_batch_size", nargs="?", type=int)
    parser.add_argument("--lang-pair")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        "--model-dir",
        dest="model_dir",
        type=Path,
        default=Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)),
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--segment-max-new-tokens",
        type=int,
        default=DEFAULT_SEGMENT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()
    args.lang_pair = args.lang_pair or args.pos_lang_pair
    args.batch_size = args.batch_size or args.pos_batch_size or 1
    if not args.lang_pair:
        parser.error("--lang-pair is required")
    if not 1 <= args.batch_size <= MAX_DECLARED_BATCH_SIZE:
        parser.error(
            f"--batch-size must be between 1 and {MAX_DECLARED_BATCH_SIZE}"
        )
    if args.max_new_tokens <= 0 or args.segment_max_new_tokens <= 0:
        parser.error("token limits must be positive")
    return args


def main() -> None:
    # Keep every library diagnostic away from the contract output channel.
    sys.stdout = sys.stderr
    args = parse_args()
    config = load_submission_config()
    if args.lang_pair != config.lang_pair:
        raise ValueError(
            f"{config.submission_id} supports only {config.lang_pair}, "
            f"not {args.lang_pair}."
        )
    source_lines = read_input_lines(args.input)
    preflight = validate_model_checkpoint(args.model_dir, config)
    log(
        f"Loading {config.submission_id} from {args.model_dir} "
        f"format={preflight['format']} average_keep={preflight['average_keep']} "
        f"batch_size={args.batch_size}"
    )
    kv_cache_bytes = int(
        os.environ.get("VLLM_KV_CACHE_MEMORY_BYTES", DEFAULT_KV_CACHE_BYTES)
    )
    max_model_len = int(
        os.environ.get("VLLM_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN)
    )
    generator = GPTOSSGenerator(
        args.model_dir,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_model_len=max_model_len,
        kv_cache_memory_bytes=kv_cache_bytes,
    )
    outputs, diagnostics = translate_lines(
        generator,
        source_lines,
        config,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        segment_max_new_tokens=args.segment_max_new_tokens,
    )
    atomic_write_lines(args.output, outputs)
    log(
        "ESTS inference summary: "
        + json.dumps(
            {
                "submission_id": config.submission_id,
                "input_lines": len(source_lines),
                "output_lines": len(outputs),
                "diagnostics": diagnostics,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
