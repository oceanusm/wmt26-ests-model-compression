#!/usr/bin/env python3
"""Pure helpers for robust WMT26 blindset validation and source-owned reconstruction."""

from __future__ import annotations

from collections import Counter
import html
import math
import re
from typing import Any, Mapping


P_BLOCK_RE = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
SUPPORTED_TAG_RE = re.compile(r"<\s*(/?)\s*(p|br)\s*/?\s*>", re.IGNORECASE)
ANY_TAG_RE = re.compile(r"<[^>]+>")
CODE_FENCE_RE = re.compile(r"\A\s*```(?:html)?\s*\n?(.*?)\n?```\s*\Z", re.I | re.S)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…。！？])\s+")


class WMT26RobustError(ValueError):
    """Raised when an output cannot be safely accepted or reconstructed."""


def strip_code_fence(text: str) -> tuple[str, bool]:
    text = str(text or "").strip()
    match = CODE_FENCE_RE.fullmatch(text)
    if match:
        return match.group(1).strip(), True
    return text, False


def canonicalize_html(text: str) -> tuple[str, list[str]]:
    text, stripped_fence = strip_code_fence(text)
    repairs = ["html_code_fence_removed"] if stripped_fence else []
    canonical = re.sub(r"<\s*p\s*>", "<p>", text, flags=re.IGNORECASE)
    canonical = re.sub(r"<\s*/\s*p\s*>", "</p>", canonical, flags=re.IGNORECASE)
    canonical = re.sub(r"<\s*br\s*/?\s*>", "<br>", canonical, flags=re.IGNORECASE)
    if canonical != text:
        repairs.append("html_tag_spelling_canonicalized")
    return canonical.strip(), repairs


def html_topology(text: str) -> list[str]:
    result = []
    for slash, tag in SUPPORTED_TAG_RE.findall(text or ""):
        normalized = tag.lower()
        result.append(f"/{normalized}" if slash else normalized)
    return result


def unsupported_html_tags(text: str, *, allow_anchor: bool = False) -> list[str]:
    unsupported = []
    for match in ANY_TAG_RE.finditer(text or ""):
        tag = match.group(0)
        if SUPPORTED_TAG_RE.fullmatch(tag):
            continue
        if allow_anchor and re.fullmatch(r"</?a(?:\s+[^>]*)?>", tag, re.IGNORECASE):
            continue
        unsupported.append(tag)
    return unsupported


def extract_exact_paragraph_blocks(
    source_html: str,
    output_html: str,
    *,
    allow_anchor: bool = False,
    allow_single_raw: bool = True,
) -> tuple[str, list[str], dict[str, Any]]:
    source, _ = canonicalize_html(source_html)
    output, repairs = canonicalize_html(output_html)
    source_topology = html_topology(source)
    output_topology = html_topology(output)
    wrapped_raw = False
    if (
        allow_single_raw
        and source_topology == ["p", "/p"]
        and not output_topology
        and not ANY_TAG_RE.search(output)
        and output.strip()
    ):
        output = f"<p>{output.strip()}</p>"
        output_topology = ["p", "/p"]
        repairs.append("single_raw_output_wrapped_as_paragraph")
        wrapped_raw = True
    if source_topology != output_topology:
        raise WMT26RobustError(
            f"HTML topology differs: expected={source_topology}, got={output_topology}."
        )
    unexpected = unsupported_html_tags(output, allow_anchor=allow_anchor)
    if unexpected:
        raise WMT26RobustError(f"Output contains unsupported HTML tags: {unexpected}.")
    blocks = P_BLOCK_RE.findall(output)
    residual = P_BLOCK_RE.sub("", output)
    if residual.strip():
        raise WMT26RobustError(
            "Output contains text or markup outside the expected paragraph blocks."
        )
    return output, blocks, {
        "repairs": repairs,
        "source_topology": source_topology,
        "output_topology": output_topology,
        "wrapped_raw": wrapped_raw,
    }


def repetition_diagnostics(text: str) -> dict[str, Any]:
    text = str(text or "")
    compact_length = len(re.sub(r"\s+", "", text))
    if compact_length == 0:
        return {"dominant_repetition": False, "kind": None, "fraction": 0.0}

    normalized_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    run_value = None
    run_count = 0
    best_run_count = 0
    best_run_chars = 0
    for line in normalized_lines:
        if not line:
            continue
        if line == run_value:
            run_count += 1
        else:
            run_value = line
            run_count = 1
        if run_count > best_run_count:
            best_run_count = run_count
            best_run_chars = len(re.sub(r"\s+", "", line)) * run_count
    line_fraction = best_run_chars / compact_length
    if best_run_count >= 4 and line_fraction >= 0.5:
        return {
            "dominant_repetition": True,
            "kind": "repeated_line",
            "fraction": line_fraction,
            "repeat_count": best_run_count,
        }

    char_runs = list(re.finditer(r"(.)\1{127,}", text, re.DOTALL))
    if char_runs:
        longest = max(char_runs, key=lambda match: len(match.group(0)))
        fraction = len(re.sub(r"\s+", "", longest.group(0))) / compact_length
        if fraction >= 0.5:
            return {
                "dominant_repetition": True,
                "kind": "repeated_character",
                "fraction": fraction,
                "repeat_count": len(longest.group(0)),
            }

    repeated_block = re.search(r"(.{32,256}?)\1{3,}", text, re.DOTALL)
    if repeated_block:
        repeated_chars = len(re.sub(r"\s+", "", repeated_block.group(0)))
        fraction = repeated_chars / compact_length
        if fraction >= 0.5:
            unit = repeated_block.group(1)
            return {
                "dominant_repetition": True,
                "kind": "repeated_substring",
                "fraction": fraction,
                "repeat_count": len(repeated_block.group(0)) // len(unit),
            }
    return {"dominant_repetition": False, "kind": None, "fraction": 0.0}


def generation_failure_reasons(
    model_hyp_text: str,
    *,
    completion_token_count: int,
    max_tokens: int,
) -> tuple[list[str], dict[str, Any]]:
    text = str(model_hyp_text or "")
    repetition = repetition_diagnostics(text)
    reasons = []
    if completion_token_count >= max_tokens:
        reasons.append("hit_max_tokens")
    if not text.strip():
        reasons.append("empty_output")
    if text.startswith("ERROR:"):
        reasons.append("harmony_parse_error")
    if repetition["dominant_repetition"]:
        reasons.append("dominant_repetition")
    return reasons, repetition


def validate_non_json_document(
    source_text: str,
    model_hyp_text: str,
    *,
    category: str,
) -> tuple[str, dict[str, Any]]:
    if category in {"news", "social"}:
        canonical, _, metadata = extract_exact_paragraph_blocks(
            source_text,
            model_hyp_text,
            allow_single_raw=True,
        )
        return canonical, metadata
    return str(model_hyp_text).strip(), {"repairs": [], "source_topology": []}


def _balanced_word_chunks(text: str, *, target_chars: int, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [text] if text else []
    desired_chunks = max(1, math.ceil(len(text) / target_chars))
    target_words = max(1, math.ceil(len(words) / desired_chunks))
    chunks = []
    cursor = 0
    while cursor < len(words):
        end = min(len(words), cursor + target_words)
        while end > cursor + 1 and len(" ".join(words[cursor:end])) > max_chars:
            end -= 1
        if end == cursor:
            end += 1
        chunks.append(" ".join(words[cursor:end]))
        cursor = end
    return chunks


def split_long_text(
    text: str,
    *,
    target_chars: int = 300,
    max_chars: int = 400,
) -> list[str]:
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return _balanced_word_chunks(text, target_chars=target_chars, max_chars=max_chars)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if current and len(candidate) > max_chars:
            chunks.extend(
                _balanced_word_chunks(current, target_chars=target_chars, max_chars=max_chars)
                if len(current) > max_chars
                else [current]
            )
            current = sentence
        else:
            current = candidate
    if current:
        chunks.extend(
            _balanced_word_chunks(current, target_chars=target_chars, max_chars=max_chars)
            if len(current) > max_chars
            else [current]
        )
    return chunks


def build_non_json_segmentation_plan(source_text: str, *, category: str) -> dict[str, Any]:
    """Build coarse units and deterministic fine units without translating anything."""
    if category in {"news", "social"}:
        source, _ = canonicalize_html(source_text)
        unexpected = unsupported_html_tags(source)
        if unexpected:
            raise WMT26RobustError(f"Source contains unsupported HTML tags: {unexpected}.")
        blocks = P_BLOCK_RE.findall(source)
        residual = P_BLOCK_RE.sub("", source)
        if not blocks or residual.strip():
            raise WMT26RobustError("Structured source is not a pure sequence of paragraphs.")
        containers = []
        for paragraph_index, block in enumerate(blocks):
            lines = BR_RE.split(block)
            fine = []
            for line_index, line in enumerate(lines):
                spans = split_long_text(line)
                fine.append(
                    {
                        "line_index": line_index,
                        "spans": [
                            {
                                "segment_id": f"p{paragraph_index}.l{line_index}.s{span_index}",
                                "model_source_text": f"<p>{span}</p>",
                                "source_text": span,
                                "wrapper": "paragraph",
                            }
                            for span_index, span in enumerate(spans)
                            if span.strip()
                        ],
                        "blank": not line.strip(),
                    }
                )
            containers.append(
                {
                    "container_id": f"p{paragraph_index}",
                    "kind": "paragraph",
                    "coarse": {
                        "segment_id": f"p{paragraph_index}",
                        "model_source_text": f"<p>{block}</p>",
                        "source_text": block,
                        "wrapper": "paragraph",
                    },
                    "fine": fine,
                }
            )
        return {"category": category, "kind": "html", "containers": containers}

    lines = source_text.split("\n")
    containers = []
    for line_index, line in enumerate(lines):
        spans = split_long_text(line)
        containers.append(
            {
                "container_id": f"l{line_index}",
                "kind": "line",
                "blank": not line.strip(),
                "coarse": None
                if not line.strip()
                else {
                    "segment_id": f"l{line_index}",
                    "model_source_text": f"<p>{html.escape(line, quote=False)}</p>",
                    "source_text": line,
                    "wrapper": "paragraph_plain",
                },
                "fine": [
                    {
                        "segment_id": f"l{line_index}.s{span_index}",
                        "model_source_text": f"<p>{html.escape(span, quote=False)}</p>",
                        "source_text": span,
                        "wrapper": "paragraph_plain",
                    }
                    for span_index, span in enumerate(spans)
                    if span.strip()
                ],
            }
        )
    return {"category": category, "kind": "lines", "containers": containers}


def validate_segment_output(
    segment: Mapping[str, Any],
    model_hyp_text: str,
) -> tuple[str, dict[str, Any]]:
    wrapper = segment.get("wrapper")
    if wrapper in {"paragraph", "paragraph_plain"}:
        _, blocks, metadata = extract_exact_paragraph_blocks(
            str(segment["model_source_text"]),
            model_hyp_text,
            allow_single_raw=True,
        )
        if len(blocks) != 1:
            raise WMT26RobustError(f"Expected one translated paragraph, got {len(blocks)}.")
        value = blocks[0].strip()
        if wrapper == "paragraph_plain":
            value = html.unescape(value)
        return value, metadata
    output, stripped = strip_code_fence(model_hyp_text)
    if ANY_TAG_RE.search(output):
        raise WMT26RobustError("Raw segment output unexpectedly contains HTML tags.")
    if "\n" in output or "\r" in output:
        raise WMT26RobustError("Raw line segment output unexpectedly contains newlines.")
    return output.strip(), {"repairs": ["code_fence_removed"] if stripped else []}


def reconstruct_non_json_segments(
    plan: Mapping[str, Any],
    translations: Mapping[str, str],
    *,
    target_lang: str,
) -> str:
    joiner = "" if target_lang == "zho_Hans" else " "
    rendered_containers = []
    for container in plan["containers"]:
        coarse = container.get("coarse")
        if coarse and coarse["segment_id"] in translations:
            rendered = translations[coarse["segment_id"]]
        elif plan["kind"] == "html":
            rendered_lines = []
            for line in container["fine"]:
                if line.get("blank"):
                    rendered_lines.append("")
                    continue
                ids = [span["segment_id"] for span in line["spans"]]
                if any(segment_id not in translations for segment_id in ids):
                    raise WMT26RobustError(
                        f"Missing fine segment translations for {container['container_id']}."
                    )
                rendered_lines.append(joiner.join(translations[segment_id] for segment_id in ids))
            rendered = "<br>".join(rendered_lines)
        else:
            if container.get("blank"):
                rendered = ""
            else:
                ids = [span["segment_id"] for span in container["fine"]]
                if any(segment_id not in translations for segment_id in ids):
                    raise WMT26RobustError(
                        f"Missing fine segment translations for {container['container_id']}."
                    )
                rendered = joiner.join(translations[segment_id] for segment_id in ids)
        rendered_containers.append(rendered)
    if plan["kind"] == "html":
        return "\n".join(f"<p>{value}</p>" for value in rendered_containers)
    return "\n".join(rendered_containers)


def raw_fallback_rank(attempt: Mapping[str, Any]) -> tuple[Any, ...]:
    reasons = set(attempt.get("failure_reasons") or [])
    metadata = attempt.get("validation") or {}
    expected = metadata.get("source_topology") or []
    actual = metadata.get("output_topology") or []
    agreement = sum(1 for left, right in zip(expected, actual) if left == right)
    return (
        "harmony_parse_error" not in reasons,
        "empty_output" not in reasons,
        "hit_max_tokens" not in reasons,
        "dominant_repetition" not in reasons,
        agreement,
        -int(attempt.get("seed", 0)),
    )


def status_counts(examples: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str((row.get("meta") or {}).get("final_status")) for row in examples).items()))
