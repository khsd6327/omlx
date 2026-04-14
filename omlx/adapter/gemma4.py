# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 reasoning-channel output parsing and message extraction."""

from __future__ import annotations

import json
import re
from typing import Any, List

try:
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
except ImportError:
    NaiveStreamingDetokenizer = None

from ..api.utils import _PRESERVE_BOUNDARY_KEY
from .output_parser import OutputParserFinalizeResult, OutputParserTokenResult

_OPEN_MARKER = "<|channel>thought\n"
_OPEN_MARKER_PREFIX = "<|channel>"
_CLOSE_MARKER = "<channel|>"
_TURN_END_MARKER = "<turn|>"
_TOOL_RESPONSE_OPEN = "<|tool_response>"
_TOOL_RESPONSE_CLOSE = "<tool_response|>"
_THINK_OPEN = "<think>\n"
_THINK_CLOSE = "</think>\n"
_OPEN_MARKER_RE = re.compile(
    r"(?:<\|channel>)+(?:[^<\r\n]{0,24})?thought(?:\r?\n)?"
)
_CLOSE_MARKER_RE = re.compile(r"(?:<channel\|>)+")
_SUPPRESSED_MARKERS = (_TURN_END_MARKER, _TOOL_RESPONSE_OPEN, _TOOL_RESPONSE_CLOSE)


def _try_parse_json(s: str) -> Any:
    """Parse string as JSON if possible, otherwise return as-is."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s or not (s.startswith("{") or s.startswith("[")):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def extract_gemma4_messages(
    messages: List[Any],
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
) -> List[dict]:
    """Convert OpenAI-format messages to Gemma 4 chat-template format.

    The Gemma 4 chat template does not handle ``role=tool`` messages.
    Tool results must instead appear on a model-role turn as a
    ``tool_responses`` list, where each entry is::

        {"name": "<function_name>", "response": <dict_or_scalar>}

    This function:
    - Passes non-tool messages through unchanged.
    - Preserves ``tool_calls`` on assistant turns (template renders them
      as ``<|tool_call>...</tool_call|>``).
    - Folds consecutive ``role=tool`` messages that follow an assistant
      turn into a single ``{"role": "assistant", "tool_responses": [...]}``
      message, resolving function names from the preceding tool_calls by
      ``tool_call_id``.  Falls back to the raw ``tool_call_id`` as the
      name when no match is found.
    - JSON-parses tool result content into a dict/list where possible so
      the template renders structured responses correctly.

    Args:
        messages: OpenAI-format Message objects or dicts.
        max_tool_result_tokens: Maximum token count for tool results
            (truncation applied when tokenizer is provided).
        tokenizer: Tokenizer for optional truncation.

    Returns:
        List of dicts ready for ``tokenizer.apply_chat_template``.
    """
    from ..api.utils import (
        _extract_text_from_content_list,
    )  # avoid circular at module level

    processed: list[dict] = []

    # Build index of message objects as plain dicts
    raw: list[dict] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            raw.append(msg.model_dump())
        elif isinstance(msg, dict):
            raw.append(dict(msg))
        else:
            raw.append(
                {
                    "role": getattr(msg, "role", "user"),
                    "content": getattr(msg, "content", ""),
                }
            )

    i = 0
    while i < len(raw):
        msg = raw[i]
        role = msg.get("role", "user")

        if role == "developer":
            role = "system"

        if role == "tool":
            # Orphaned tool result with no preceding assistant turn — attach
            # to a synthetic assistant turn with no content.
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)
            if max_tool_result_tokens and tokenizer and content:
                from ..api.anthropic_utils import truncate_tool_result

                content = truncate_tool_result(
                    content, max_tool_result_tokens, tokenizer
                )
            response = _try_parse_json(content)
            # Fallback name: use tool_call_id
            processed.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_responses": [
                        {"name": tool_call_id or "unknown", "response": response}
                    ],
                    _PRESERVE_BOUNDARY_KEY: True,
                }
            )
            i += 1
            continue

        if role == "assistant":
            # Build a tool_call_id → function_name lookup from this turn's calls.
            tc_id_to_name: dict[str, str] = {}
            tool_calls_raw = msg.get("tool_calls") or []
            for tc in tool_calls_raw:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    func_name = (tc.get("function") or {}).get("name", "")
                else:
                    tc_id = getattr(tc, "id", "")
                    func = getattr(tc, "function", None)
                    func_name = getattr(func, "name", "") if func else ""
                if tc_id:
                    tc_id_to_name[tc_id] = func_name

            # Extract content
            content = msg.get("content", "")
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)

            out_msg: dict = {"role": "assistant", "content": content or ""}

            # Preserve tool_calls for template rendering
            if tool_calls_raw:
                out_calls = []
                for tc in tool_calls_raw:
                    if isinstance(tc, dict):
                        func = tc.get("function") or {}
                        out_calls.append(
                            {
                                "id": tc.get("id", ""),
                                "function": {
                                    "name": func.get("name", ""),
                                    "arguments": _try_parse_json(
                                        func.get("arguments", "{}")
                                    ),
                                },
                            }
                        )
                    else:
                        func = getattr(tc, "function", None)
                        args_str = getattr(func, "arguments", "{}") if func else "{}"
                        out_calls.append(
                            {
                                "id": getattr(tc, "id", ""),
                                "function": {
                                    "name": getattr(func, "name", "") if func else "",
                                    "arguments": _try_parse_json(args_str),
                                },
                            }
                        )
                out_msg["tool_calls"] = out_calls
                out_msg[_PRESERVE_BOUNDARY_KEY] = True

            processed.append(out_msg)
            i += 1

            # Consume any immediately following tool results into a
            # single model turn with tool_responses.
            tool_responses = []
            while i < len(raw) and raw[i].get("role") == "tool":
                tr = raw[i]
                tc_id = tr.get("tool_call_id", "")
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = _extract_text_from_content_list(tr_content)
                if max_tool_result_tokens and tokenizer and tr_content:
                    from ..api.anthropic_utils import truncate_tool_result

                    tr_content = truncate_tool_result(
                        tr_content, max_tool_result_tokens, tokenizer
                    )
                response = _try_parse_json(tr_content)
                name = tc_id_to_name.get(tc_id) or tc_id or "unknown"
                tool_responses.append({"name": name, "response": response})
                i += 1

            if tool_responses:
                processed.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_responses": tool_responses,
                        _PRESERVE_BOUNDARY_KEY: True,
                    }
                )
            continue

        # All other roles (user, system)
        # Preserve image_url parts for VLM processing
        content = msg.get("content", "")
        if isinstance(content, list):
            from ..api.utils import _extract_multimodal_content_list

            multimodal_parts = _extract_multimodal_content_list(content)
            has_images = any(p.get("type") == "image_url" for p in multimodal_parts)
            if has_images:
                content = multimodal_parts
            else:
                content = _extract_text_from_content_list(content)
        out: dict = {"role": role, "content": content if content is not None else ""}
        processed.append(out)
        i += 1

    # Standard cleanup passes shared with other extractors
    from ..api.utils import (
        _consolidate_system_messages,
        _drop_void_assistant_messages,
        _merge_consecutive_roles,
    )

    return _merge_consecutive_roles(
        _drop_void_assistant_messages(_consolidate_system_messages(processed))
    )


def _matching_prefix_len(text: str, marker: str) -> int:
    """Return longest suffix of ``text`` that is a prefix of ``marker``."""
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


def _find_open_candidate_start(text: str) -> int | None:
    """Return the first likely start of a Gemma reasoning channel marker."""
    idx = text.find(_OPEN_MARKER_PREFIX)
    if idx != -1:
        return idx

    last_lt = text.rfind("<")
    if last_lt == -1:
        return None

    candidate = text[last_lt:]
    if _OPEN_MARKER_PREFIX.startswith(candidate):
        return last_lt
    return None


class Gemma4ReasoningTextNormalizer:
    """Normalize Gemma 4 channel markers into ``<think>`` tags.

    Handles the standard marker as well as malformed variants seen in the
    wild, such as repeated ``<|channel>`` prefixes or numeric junk before
    ``thought`` (for example ``<|channel><|channel>42thought``).
    """

    def __init__(self):
        self._buffer = ""
        self._in_thought = False
        self._suppress_next_leading_newline = False

    def _find_next_marker(
        self, source: str, pos: int
    ) -> tuple[int, int, str] | tuple[None, None, None]:
        candidates: list[tuple[int, int, str]] = []

        open_match = _OPEN_MARKER_RE.search(source, pos)
        if open_match:
            candidates.append((open_match.start(), open_match.end(), "open"))

        close_match = _CLOSE_MARKER_RE.search(source, pos)
        if close_match:
            candidates.append((close_match.start(), close_match.end(), "close"))

        for marker in _SUPPRESSED_MARKERS:
            idx = source.find(marker, pos)
            if idx != -1:
                candidates.append((idx, idx + len(marker), "suppress"))

        if not candidates:
            return None, None, None
        return min(candidates, key=lambda item: item[0])

    def _partial_suffix_len(self, text: str) -> int:
        keep = 0
        for marker in (_CLOSE_MARKER, *_SUPPRESSED_MARKERS):
            keep = max(keep, _matching_prefix_len(text, marker))

        open_candidate_start = _find_open_candidate_start(text)
        if open_candidate_start is not None:
            keep = max(keep, len(text) - open_candidate_start)

        return min(keep, 128)

    def _append_text(self, out: list[str], text: str) -> None:
        if not text:
            return
        if self._suppress_next_leading_newline:
            if text.startswith("\r\n"):
                text = text[2:]
            elif text.startswith("\n"):
                text = text[1:]
            self._suppress_next_leading_newline = False
            if not text:
                return
        out.append(text)

    def _consume_text(self, text: str, *, final: bool = False) -> str:
        source = self._buffer + text
        self._buffer = ""

        out: list[str] = []
        pos = 0

        while pos < len(source):
            start, end, kind = self._find_next_marker(source, pos)

            if start is None or end is None or kind is None:
                remainder = source[pos:]
                if not final:
                    open_candidate_start = _find_open_candidate_start(remainder)
                    if open_candidate_start is not None:
                        self._append_text(out, remainder[:open_candidate_start])
                        self._buffer = remainder[open_candidate_start:]
                        break

                    keep = self._partial_suffix_len(remainder)
                    if keep:
                        self._append_text(out, remainder[:-keep])
                        self._buffer = remainder[-keep:]
                    else:
                        self._append_text(out, remainder)
                else:
                    open_candidate_start = _find_open_candidate_start(remainder)
                    if open_candidate_start is not None:
                        self._append_text(out, remainder[:open_candidate_start])
                    else:
                        keep = self._partial_suffix_len(remainder)
                        self._append_text(out, remainder[:-keep] if keep else remainder)
                break

            if start > pos:
                self._append_text(out, source[pos:start])

            if kind == "open":
                if not self._in_thought:
                    out.append(_THINK_OPEN)
                    self._in_thought = True
                    self._suppress_next_leading_newline = not source[start:end].endswith("\n")
            elif kind == "close":
                if self._in_thought:
                    out.append(_THINK_CLOSE)
                    self._in_thought = False
                self._suppress_next_leading_newline = False

            pos = end

        return "".join(out)

    def feed(self, text: str) -> str:
        """Feed a text chunk and return normalized text safe to emit."""
        if not text:
            return ""
        return self._consume_text(text)

    def finish(self) -> str:
        """Flush any buffered marker fragments at end-of-stream."""
        out = self._consume_text("", final=True)
        if self._buffer:
            out += self._buffer
            self._buffer = ""
        if self._in_thought:
            out += _THINK_CLOSE
            self._in_thought = False
        return out


def normalize_gemma4_reasoning_text(text: str) -> str:
    """Normalize Gemma 4 reasoning-channel protocol markers in complete text."""
    normalizer = Gemma4ReasoningTextNormalizer()
    return normalizer.feed(text) + normalizer.finish()


class Gemma4OutputParserSession:
    """Suppress Gemma 4 protocol markers and re-emit thought blocks as ``<think>`` tags."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._normalizer = Gemma4ReasoningTextNormalizer()

        if hasattr(tokenizer, "detokenizer"):
            self._detokenizer = tokenizer.detokenizer
        elif NaiveStreamingDetokenizer is not None:
            self._detokenizer = NaiveStreamingDetokenizer(tokenizer)
        else:
            self._detokenizer = None

        if self._detokenizer is not None:
            self._detokenizer.reset()

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            text = self._detokenizer.last_segment
        else:
            text = self._tokenizer.decode([token_id])
        normalized = self._normalizer.feed(text)
        return OutputParserTokenResult(
            stream_text=normalized,
            visible_text=normalized,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        text = ""
        if self._detokenizer is not None:
            self._detokenizer.finalize()
            text = self._detokenizer.last_segment

        stream_text = self._normalizer.feed(text)
        stream_text += self._normalizer.finish()

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=stream_text,
        )
