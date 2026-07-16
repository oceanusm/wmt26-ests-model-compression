#!/usr/bin/env python3
"""Reversible adapters for WMT26 software-data JSON translation rows."""

from __future__ import annotations

from copy import deepcopy
import html
import json
import re
from typing import Any, Iterable, Mapping


JSON_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
P_BLOCK_RE = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
PLACEHOLDER_RE = re.compile(
    r"({{[^{}]+}}|\$\{[^{}]+\}|%\([^)]+\)[sd]|%[sdif]|"
    r"\{[A-Za-z0-9_.-]+\}|\$[A-Za-z_][A-Za-z0-9_]*)"
)
ROBUST_PLACEHOLDER_CANDIDATE_RE = re.compile(
    r"({{[^{}\n]+}}|\$\{[^{}\n]+\}|%\([^)\n]+\)[sd]|%[sdif]|"
    r"\{[^{}\n]+\}|\$[A-Za-z_][A-Za-z0-9_]*)"
)
SENTINEL_RE = re.compile(
    r"ZXQPH[\s_-]*([0-9]{6})[\s_-]*QXZ",
    re.IGNORECASE,
)
ANCHOR_START_RE = re.compile(r"<a\b[^>]*>", re.IGNORECASE)
ANCHOR_END_RE = re.compile(r"</a\s*>", re.IGNORECASE)


class WMT26JSONError(ValueError):
    """Raised when a WMT26 JSON row cannot be projected or reconstructed."""


def _strip_json_fence(text: str) -> tuple[str, bool]:
    stripped = str(text).strip()
    match = JSON_FENCE_RE.fullmatch(stripped)
    if match:
        return match.group(1).strip(), True
    return stripped, False


def _strip_trailing_commas(text: str) -> tuple[str, int]:
    """Remove commas immediately before ] or } while respecting JSON strings."""
    out: list[str] = []
    in_string = False
    escaped = False
    removed = 0
    idx = 0
    while idx < len(text):
        char = text[idx]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            idx += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            idx += 1
            continue

        if char == ",":
            lookahead = idx + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "]}":
                removed += 1
                idx += 1
                continue

        out.append(char)
        idx += 1
    return "".join(out), removed


def parse_wmt26_json_source(source_text: str) -> tuple[Any, dict[str, Any]]:
    body, fenced = _strip_json_fence(source_text)
    cleaned, removed_trailing_commas = _strip_trailing_commas(body)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise WMT26JSONError(f"Could not parse WMT26 JSON source: {exc}") from exc
    return value, {
        "source_was_fenced": fenced,
        "source_trailing_commas_removed": removed_trailing_commas,
    }


def parse_json_output(output_text: str) -> Any:
    body, _ = _strip_json_fence(output_text)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise WMT26JSONError(f"Model output is not valid JSON: {exc}") from exc


def _iter_string_leaves(value: Any, path: tuple[Any, ...] = ()) -> Iterable[tuple[tuple[Any, ...], str]]:
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_string_leaves(child, path + (key,))
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _iter_string_leaves(child, path + (idx,))


def _set_path(value: Any, path: tuple[Any, ...], replacement: str) -> None:
    if not path:
        raise WMT26JSONError("A top-level JSON string is not supported by the projection adapter.")
    target = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement


def _placeholder_list(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text or "")


def _compare_structure(source: Any, translated: Any, path: str = "$") -> None:
    if isinstance(source, dict):
        if not isinstance(translated, dict):
            raise WMT26JSONError(f"Expected object at {path}, got {type(translated).__name__}.")
        if list(translated) != list(source):
            if set(translated) != set(source):
                missing = sorted(set(source) - set(translated))
                extra = sorted(set(translated) - set(source))
                raise WMT26JSONError(f"JSON keys differ at {path}: missing={missing}, extra={extra}.")
        for key, child in source.items():
            _compare_structure(child, translated[key], f"{path}.{key}")
        return
    if isinstance(source, list):
        if not isinstance(translated, list):
            raise WMT26JSONError(f"Expected array at {path}, got {type(translated).__name__}.")
        if len(translated) != len(source):
            raise WMT26JSONError(
                f"Array length differs at {path}: expected={len(source)}, got={len(translated)}."
            )
        for idx, (source_child, translated_child) in enumerate(zip(source, translated)):
            _compare_structure(source_child, translated_child, f"{path}[{idx}]")
        return
    if isinstance(source, str):
        if not isinstance(translated, str):
            raise WMT26JSONError(f"Expected string at {path}, got {type(translated).__name__}.")
        source_placeholders = _placeholder_list(source)
        translated_placeholders = _placeholder_list(translated)
        if source_placeholders != translated_placeholders:
            raise WMT26JSONError(
                f"Placeholders differ at {path}: expected={source_placeholders}, "
                f"got={translated_placeholders}."
            )
        return
    if translated != source or type(translated) is not type(source):
        raise WMT26JSONError(
            f"Non-string value changed at {path}: expected={source!r}, got={translated!r}."
        )


def _reorder_like_source(source: Any, translated: Any) -> Any:
    if isinstance(source, dict):
        return {key: _reorder_like_source(value, translated[key]) for key, value in source.items()}
    if isinstance(source, list):
        return [
            _reorder_like_source(source_child, translated_child)
            for source_child, translated_child in zip(source, translated)
        ]
    return translated


def render_json(value: Any, *, include_markdown_fence: bool = False) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    if include_markdown_fence:
        return f"```json\n{rendered}\n```"
    return rendered


def validate_and_canonicalize_json_translation(
    source_text: str,
    output_text: str,
    *,
    unwrap_translation: bool = False,
) -> tuple[str, dict[str, Any]]:
    source, parse_meta = parse_wmt26_json_source(source_text)
    translated = parse_json_output(output_text)
    if unwrap_translation:
        if not isinstance(translated, dict) or set(translated) != {"translation"}:
            raise WMT26JSONError("Expected a structured-output wrapper with only `translation`.")
        translated = translated["translation"]
    _compare_structure(source, translated)
    translated = _reorder_like_source(source, translated)
    leaves = list(_iter_string_leaves(source))
    canonical = render_json(translated)
    return canonical, {
        **parse_meta,
        "valid": True,
        "root_type": "object" if isinstance(source, dict) else "array" if isinstance(source, list) else type(source).__name__,
        "string_leaf_count": len(leaves),
        "placeholder_count": sum(len(_placeholder_list(text)) for _, text in leaves),
        "structured_output_wrapper_removed": unwrap_translation,
    }


def _schema_for_value(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, bool):
        return {"type": "boolean", "enum": [value]}
    if value is None:
        return {"type": "null"}
    if isinstance(value, int):
        return {"type": "integer", "enum": [value]}
    if isinstance(value, float):
        return {"type": "number", "enum": [value]}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: _schema_for_value(child) for key, child in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        if not value:
            return {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 0}
        item_schemas = [_schema_for_value(child) for child in value]
        if any(schema != item_schemas[0] for schema in item_schemas[1:]):
            raise WMT26JSONError("Heterogeneous JSON arrays are not supported by strict output schemas.")
        return {
            "type": "array",
            "items": item_schemas[0],
            "minItems": len(value),
            "maxItems": len(value),
        }
    raise WMT26JSONError(f"Unsupported JSON value type: {type(value).__name__}")


def build_openai_json_text_format(source_text: str) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    source, parse_meta = parse_wmt26_json_source(source_text)
    schema = _schema_for_value(source)
    wrapped = not isinstance(source, dict)
    if wrapped:
        schema = {
            "type": "object",
            "properties": {"translation": schema},
            "required": ["translation"],
            "additionalProperties": False,
        }
    return {
        "type": "json_schema",
        "name": "wmt26_json_translation",
        "strict": True,
        "schema": schema,
    }, wrapped, parse_meta


def project_json_to_html(
    source_text: str,
    model_instruction: str,
) -> tuple[str, str, dict[str, Any]]:
    """Render JSON string leaves as paragraphs under a caller-selected instruction."""
    model_instruction = str(model_instruction).strip()
    if not model_instruction:
        raise WMT26JSONError("JSON HTML projection requires a non-empty model instruction.")
    source, parse_meta = parse_wmt26_json_source(source_text)
    leaves = list(_iter_string_leaves(source))
    if not leaves:
        raise WMT26JSONError("JSON source has no translatable string leaves.")
    blocks = []
    for _, text in leaves:
        escaped = html.escape(text, quote=False).replace("\n", "<br>")
        blocks.append(f"<p>{escaped}</p>")
    projected_source = "\n".join(blocks)
    return projected_source, model_instruction, {
        **parse_meta,
        "projection": "json_string_leaves_to_html_news_prompt_v2",
        "string_leaf_count": len(leaves),
        "placeholder_count": sum(len(_placeholder_list(text)) for _, text in leaves),
    }


def reconstruct_json_from_html(source_text: str, translated_html: str) -> tuple[str, dict[str, Any]]:
    source, parse_meta = parse_wmt26_json_source(source_text)
    leaves = list(_iter_string_leaves(source))
    translated_html = str(translated_html)
    blocks = P_BLOCK_RE.findall(translated_html)
    if len(blocks) != len(leaves):
        raise WMT26JSONError(
            f"Projected paragraph count differs: expected={len(leaves)}, got={len(blocks)}."
        )
    residual = P_BLOCK_RE.sub("", translated_html)
    if residual.strip():
        raise WMT26JSONError(
            "Projected output contains text or markup outside the expected paragraph blocks."
        )

    reconstructed = deepcopy(source)
    for (path, source_value), block in zip(leaves, blocks):
        block_without_breaks = BR_RE.sub("\n", block)
        unexpected_tags = HTML_TAG_RE.findall(block_without_breaks)
        if unexpected_tags:
            raise WMT26JSONError(
                f"Projected output contains unsupported HTML tags at {path}: {unexpected_tags}."
            )
        translated_value = html.unescape(block_without_breaks).strip()
        source_placeholders = _placeholder_list(source_value)
        translated_placeholders = _placeholder_list(translated_value)
        if source_placeholders != translated_placeholders:
            raise WMT26JSONError(
                f"Placeholders differ at {path}: expected={source_placeholders}, "
                f"got={translated_placeholders}."
            )
        _set_path(reconstructed, path, translated_value)

    _compare_structure(source, reconstructed)
    canonical = render_json(reconstructed)
    return canonical, {
        **parse_meta,
        "valid": True,
        "projection": "json_string_leaves_to_html_news_prompt_v2",
        "strict_projection_html": True,
        "string_leaf_count": len(leaves),
        "placeholder_count": sum(len(_placeholder_list(text)) for _, text in leaves),
    }


def sentinel_for_index(index: int) -> str:
    if not 0 <= int(index) <= 999999:
        raise WMT26JSONError(f"Sentinel index is out of range: {index}")
    return f"ZXQPH{int(index):06d}QXZ"


def _path_to_json(path: tuple[Any, ...]) -> list[Any]:
    return list(path)


def _mask_leaf_placeholders(
    text: str,
    *,
    start_index: int,
) -> tuple[str, list[dict[str, Any]], int]:
    placeholder_records: list[dict[str, Any]] = []
    next_index = int(start_index)

    def replace(match: re.Match[str]) -> str:
        nonlocal next_index
        marker = sentinel_for_index(next_index)
        placeholder_records.append(
            {
                "index": next_index,
                "marker": marker,
                "placeholder": match.group(0),
            }
        )
        next_index += 1
        return marker

    return PLACEHOLDER_RE.sub(replace, text), placeholder_records, next_index


def build_robust_json_leaves(
    source_text: str,
    *,
    placeholder_mode: str,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    if placeholder_mode not in {"raw-repair", "sentinel-repair"}:
        raise WMT26JSONError(f"Unsupported robust placeholder mode: {placeholder_mode!r}")
    source, parse_meta = parse_wmt26_json_source(source_text)
    leaves = list(_iter_string_leaves(source))
    if not leaves:
        raise WMT26JSONError("JSON source has no translatable string leaves.")

    result = []
    next_sentinel_index = 0
    for leaf_index, (path, source_value) in enumerate(leaves):
        model_value = source_value
        placeholder_records: list[dict[str, Any]] = []
        if placeholder_mode == "sentinel-repair":
            model_value, placeholder_records, next_sentinel_index = _mask_leaf_placeholders(
                source_value,
                start_index=next_sentinel_index,
            )
        result.append(
            {
                "leaf_index": leaf_index,
                "path": _path_to_json(path),
                "source_value": source_value,
                "model_value": model_value,
                "source_placeholders": _placeholder_list(source_value),
                "placeholder_records": placeholder_records,
                "projected_html": (
                    "<p>"
                    + html.escape(model_value, quote=False).replace("\n", "<br>")
                    + "</p>"
                ),
            }
        )
    return source, result, parse_meta


def project_json_to_html_robust(
    source_text: str,
    model_instruction: str,
    *,
    placeholder_mode: str,
) -> tuple[str, str, dict[str, Any]]:
    """Project JSON leaves for robust inference without changing the legacy adapter."""
    model_instruction = str(model_instruction).strip()
    if not model_instruction:
        raise WMT26JSONError("JSON HTML projection requires a non-empty model instruction.")
    _, leaves, parse_meta = build_robust_json_leaves(
        source_text,
        placeholder_mode=placeholder_mode,
    )
    projected_source = "\n".join(str(leaf["projected_html"]) for leaf in leaves)
    return projected_source, model_instruction, {
        **parse_meta,
        "projection": "json_string_leaves_to_html_news_prompt_robust_v1",
        "placeholder_mode": placeholder_mode,
        "string_leaf_count": len(leaves),
        "placeholder_count": sum(len(leaf["source_placeholders"]) for leaf in leaves),
        "leaves": leaves,
    }


def _repair_anchor_placeholders(text: str, expected: list[str]) -> tuple[str, list[str]]:
    repairs: list[str] = []
    if "{{link_tag_start}}" not in expected or "{{link_tag_end}}" not in expected:
        return text, repairs
    if len(ANCHOR_START_RE.findall(text)) != 1 or len(ANCHOR_END_RE.findall(text)) != 1:
        return text, repairs
    text = ANCHOR_START_RE.sub("{{link_tag_start}}", text, count=1)
    text = ANCHOR_END_RE.sub("{{link_tag_end}}", text, count=1)
    repairs.append("html_anchor_to_link_placeholders")
    return text, repairs


def _repair_raw_placeholders(text: str, expected: list[str]) -> tuple[str, list[str]]:
    if _placeholder_list(text) == expected:
        return text, []
    candidates = list(ROBUST_PLACEHOLDER_CANDIDATE_RE.finditer(text))
    if len(candidates) != len(expected):
        raise WMT26JSONError(
            "Raw placeholder candidate count differs: "
            f"expected={len(expected)}, got={len(candidates)}."
        )
    out: list[str] = []
    cursor = 0
    for candidate, replacement in zip(candidates, expected):
        out.append(text[cursor : candidate.start()])
        out.append(replacement)
        cursor = candidate.end()
    out.append(text[cursor:])
    repaired = "".join(out)
    if _placeholder_list(repaired) != expected:
        raise WMT26JSONError(
            f"Raw placeholder positional repair failed: expected={expected}, "
            f"got={_placeholder_list(repaired)}."
        )
    return repaired, ["raw_placeholders_replaced_positionally"]


def _restore_sentinels(
    text: str,
    placeholder_records: list[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    expected_indices = [int(record["index"]) for record in placeholder_records]
    seen_matches = list(SENTINEL_RE.finditer(text))
    seen_indices = [int(match.group(1)) for match in seen_matches]
    if not expected_indices and not seen_indices:
        return text, []
    if seen_indices != expected_indices:
        raise WMT26JSONError(
            f"Sentinels differ: expected={expected_indices}, got={seen_indices}."
        )
    by_index = {
        int(record["index"]): str(record["placeholder"])
        for record in placeholder_records
    }
    canonicalized = any(
        match.group(0) != sentinel_for_index(int(match.group(1)))
        for match in seen_matches
    )
    restored = SENTINEL_RE.sub(
        lambda match: by_index[int(match.group(1))],
        text,
    )
    repairs = ["sentinel_spelling_canonicalized"] if canonicalized else []
    repairs.append("sentinels_restored")
    return restored, repairs


def validate_robust_json_leaf_translation(
    leaf: Mapping[str, Any],
    translated_value: str,
    *,
    placeholder_mode: str,
) -> tuple[str, list[str]]:
    """Restore immutable markers in one decoded leaf and validate exact placeholders."""
    translated_value = str(translated_value).strip()
    expected = [str(value) for value in leaf.get("source_placeholders", [])]
    repairs: list[str] = []
    if placeholder_mode == "sentinel-repair":
        translated_value, marker_repairs = _restore_sentinels(
            translated_value,
            list(leaf.get("placeholder_records", [])),
        )
        repairs.extend(marker_repairs)
    elif placeholder_mode == "raw-repair":
        translated_value, anchor_repairs = _repair_anchor_placeholders(
            translated_value,
            expected,
        )
        repairs.extend(anchor_repairs)
        translated_value, placeholder_repairs = _repair_raw_placeholders(
            translated_value,
            expected,
        )
        repairs.extend(placeholder_repairs)
    else:
        raise WMT26JSONError(f"Unsupported robust placeholder mode: {placeholder_mode!r}")
    actual = _placeholder_list(translated_value)
    if actual != expected:
        raise WMT26JSONError(f"Placeholders differ: expected={expected}, got={actual}.")
    return translated_value, repairs


def reconstruct_json_from_robust_leaf_values(
    source_text: str,
    translated_values: Mapping[int, str],
) -> tuple[str, dict[str, Any]]:
    source, parse_meta = parse_wmt26_json_source(source_text)
    leaves = list(_iter_string_leaves(source))
    reconstructed = deepcopy(source)
    missing = []
    for leaf_index, (path, _) in enumerate(leaves):
        if leaf_index not in translated_values:
            missing.append(leaf_index)
            continue
        _set_path(reconstructed, path, str(translated_values[leaf_index]))
    if missing:
        raise WMT26JSONError(f"Missing translated JSON leaves: {missing}")
    _compare_structure(source, reconstructed)
    return render_json(reconstructed), {
        **parse_meta,
        "valid": True,
        "projection": "json_string_leaves_to_html_news_prompt_robust_v1",
        "string_leaf_count": len(leaves),
        "placeholder_count": sum(len(_placeholder_list(text)) for _, text in leaves),
    }


def reconstruct_json_from_html_robust(
    source_text: str,
    translated_blocks: list[str],
    *,
    projection_meta: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Reconstruct robust projected blocks after the caller validates HTML topology."""
    leaves = list(projection_meta.get("leaves", []))
    if len(translated_blocks) != len(leaves):
        raise WMT26JSONError(
            f"Projected paragraph count differs: expected={len(leaves)}, "
            f"got={len(translated_blocks)}."
        )
    placeholder_mode = str(projection_meta.get("placeholder_mode"))
    values: dict[int, str] = {}
    repairs: list[dict[str, Any]] = []
    for leaf, block in zip(leaves, translated_blocks):
        block_without_breaks = BR_RE.sub("\n", str(block))
        unexpected_tags = HTML_TAG_RE.findall(block_without_breaks)
        if unexpected_tags and not (
            placeholder_mode == "raw-repair"
            and "{{link_tag_start}}" in leaf.get("source_placeholders", [])
            and all(
                ANCHOR_START_RE.fullmatch(tag) or ANCHOR_END_RE.fullmatch(tag)
                for tag in unexpected_tags
            )
        ):
            raise WMT26JSONError(
                f"Projected output contains unsupported HTML tags at {leaf['path']}: "
                f"{unexpected_tags}."
            )
        translated_value = html.unescape(block_without_breaks).strip()
        translated_value, leaf_repairs = validate_robust_json_leaf_translation(
            leaf,
            translated_value,
            placeholder_mode=placeholder_mode,
        )
        leaf_index = int(leaf["leaf_index"])
        values[leaf_index] = translated_value
        if leaf_repairs:
            repairs.append({"leaf_index": leaf_index, "repairs": leaf_repairs})
    canonical, validation = reconstruct_json_from_robust_leaf_values(source_text, values)
    return canonical, {
        **validation,
        "placeholder_mode": placeholder_mode,
        "repairs": repairs,
    }


def split_json_leaf_around_placeholders(source_value: str) -> list[dict[str, str]]:
    """Return alternating translatable and immutable portions of a JSON leaf."""
    parts: list[dict[str, str]] = []
    cursor = 0
    for match in PLACEHOLDER_RE.finditer(source_value):
        if match.start() > cursor:
            parts.append({"kind": "text", "value": source_value[cursor : match.start()]})
        parts.append({"kind": "placeholder", "value": match.group(0)})
        cursor = match.end()
    if cursor < len(source_value):
        parts.append({"kind": "text", "value": source_value[cursor:]})
    if not parts:
        parts.append({"kind": "text", "value": source_value})
    return parts
