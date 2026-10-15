"""LLM inference against an OpenAI-compatible local server.

Talks to LM Studio, llama.cpp's ``llama-server``, vLLM, Ollama, or any server that
speaks ``/v1/chat/completions``. Importing this module never opens a connection and
never requires the ``openai`` package until a request is actually made.

Three call styles:

    client.chat(messages)                        -> Completion (blocking)
    client.stream_chat(messages, on_chunk=...)   -> Completion (token callback)
    await client.stream(messages)                -> async iterator of chunks
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from core.config import LLMSettings
from core.log import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ChatMessage:
    """One entry in a conversation, in the wire format servers expect."""

    role: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"role": self.role}
        # Some servers (llama.cpp, some Qwen builds) hard-fail on null content.
        out["content"] = self.content if self.content else ""
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=str(data.get("role", "user")),
            content=data.get("content") or "",
            tool_calls=list(data.get("tool_calls") or []),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )

    def is_tool_result(self) -> bool:
        return self.role == "tool"


@dataclass(slots=True)
class ToolCall:
    """A tool invocation the model requested, decoded and ready to dispatch."""

    id: str
    name: str
    arguments: dict
    raw_arguments: str

    def to_dict(self) -> dict:
        """Wire format for replaying this call into a follow-up request."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    def result_message(self, output: str) -> ChatMessage:
        """The tool-role message that closes this call in the conversation."""
        return ChatMessage(role="tool", content=output, tool_call_id=self.id, name=self.name)


@dataclass(slots=True)
class Completion:
    """A finished chat completion, with its own accounting attached."""

    text: str
    messages: list[dict]
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    elapsed_s: float = 0.0
    finish_reason: str = "stop"

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_s <= 0 or not self.completion_tokens:
            return 0.0
        return self.completion_tokens / self.elapsed_s

    @property
    def used_tokens(self) -> int:
        return self.total_tokens or (self.prompt_tokens + self.completion_tokens)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """The server cannot be reached, or returned something unusable."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class LLMClient:
    """A small, dependency-tolerant wrapper over an OpenAI-compatible endpoint."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings()
        self._client = None

    # -- transport ---------------------------------------------------------

    @property
    def client(self):
        """Lazy OpenAI client, built once per endpoint."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError(
                    "The 'openai' package is required for LLM inference. "
                    "Run:  pip install openai"
                ) from exc
            self._client = OpenAI(
                base_url=self.settings.server_url.rstrip("/"),
                api_key=self.settings.api_key or "lm-studio",
                timeout=self.settings.timeout_s,
                max_retries=0,  # We retry and report latency ourselves.
            )
        return self._client

    @property
    def is_available(self) -> bool:
        """Whether the SDK can be imported. Never raises."""
        try:
            import openai  # noqa: F401

            return True
        except ImportError:
            return False

    def ping(self, timeout_s: float = 3.0) -> bool:
        """Confirm the server is up. Never raises; safe from ``--check``."""
        url = self.settings.server_url.rstrip("/")
        try:
            import httpx
        except ImportError:
            if not self.is_available:
                return False
            try:
                self.client.models.list()
                return True
            except Exception:
                return False

        for path in ("/models", "/health", "/"):
            try:
                resp = httpx.get(url + path, timeout=timeout_s)
            except Exception:
                continue
            if resp.status_code < 500:
                return True
        return False

    def list_models(self) -> list[str]:
        """Model ids the server advertises. Empty on any failure."""
        try:
            resp = self.client.models.list()
        except Exception as exc:
            log.debug("model listing failed: %s", exc)
            return []
        return sorted(str(m.id) for m in resp.data)

    # -- request construction ------------------------------------------------

    def _base_params(self, *, stream: bool) -> dict:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
            "stream": stream,
        }
        extra = self._extra_body()
        if extra:
            params["extra_body"] = extra
        return params

    def _extra_body(self) -> dict | None:
        """Server-specific knobs. Empty dicts are dropped so servers see a clean body."""
        body = dict(self.settings.extra_body) if self.settings.extra_body else {}
        return body or None

    @staticmethod
    def _normalize(messages: list[ChatMessage | dict]) -> list[dict]:
        return [m if isinstance(m, dict) else m.to_dict() for m in messages]

    def _usage(self, resp) -> tuple[int, int, int]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return 0, 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            int(getattr(usage, "total_tokens", 0) or 0),
        )

    # -- non-streaming -------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage | dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        """A single non-streaming completion. Blocks; call from a worker thread."""
        normalized = self._normalize(messages)
        params = self._base_params(stream=False)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(messages=normalized, **params)
        except Exception as exc:
            raise _wrap(exc, self.settings.server_url) from exc
        elapsed = time.perf_counter() - started

        choice = resp.choices[0]
        message = choice.message
        prompt, completion_tokens, total = self._usage(resp)
        result = Completion(
            text=message.content or "",
            messages=normalized,
            tool_calls=_decode_tool_calls(getattr(message, "tool_calls", None)),
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            total_tokens=total,
            model=str(getattr(resp, "model", "") or ""),
            elapsed_s=elapsed,
            finish_reason=str(getattr(choice, "finish_reason", "stop") or "stop"),
        )
        _log_completion(result, extra={"stream": False})
        return result

    # -- synchronous streaming -----------------------------------------------

    def stream_chat(
        self,
        messages: list[ChatMessage | dict],
        *,
        on_chunk: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Stream tokens synchronously, calling ``on_chunk`` for each decoded piece.

        Streaming is what lets TTS start speaking before the model finishes. We
        return a fully-formed Completion so the caller has tokens-per-second and tool
        calls whether or not it consumed the stream. Tool calls are accumulated across
        chunks and decoded exactly as in :meth:`chat`.
        """
        normalized = self._normalize(messages)
        params = self._base_params(stream=True)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        started = time.perf_counter()
        try:
            stream = self.client.chat.completions.create(messages=normalized, **params)
        except Exception as exc:
            raise _wrap(exc, self.settings.server_url) from exc

        parts: list[str] = []
        tool_state: dict[int, dict] = {}
        finish_reason = "stop"
        model = ""

        for chunk in stream:
            if getattr(chunk, "model", ""):
                model = str(chunk.model)
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = str(choice.finish_reason)
            piece = getattr(choice.delta, "content", None)
            if piece:
                parts.append(piece)
                if on_chunk is not None:
                    _safe_chunk(on_chunk, piece)
            for tc in getattr(choice.delta, "tool_calls", None) or []:
                _accumulate_tool(tool_state, tc)

        elapsed = time.perf_counter() - started
        result = Completion(
            text="".join(parts),
            messages=normalized,
            tool_calls=_finalize_tools(tool_state),
            model=model,
            elapsed_s=elapsed,
            finish_reason=finish_reason,
        )
        _log_completion(result, extra={"stream": True})
        return result

    # -- async streaming -----------------------------------------------------

    async def stream(
        self,
        messages: list[ChatMessage | dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield decoded chunks asynchronously. Runs the SDK off the event loop."""
        import collections

        normalized = self._normalize(messages)
        params = self._base_params(stream=True)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        queue: collections.deque = collections.deque()
        error: list[BaseException] = []

        def _produce() -> None:
            try:
                stream = self.client.chat.completions.create(messages=normalized, **params)
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    piece = getattr(chunk.choices[0].delta, "content", None)
                    if piece:
                        queue.append(piece)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
            finally:
                queue.append(None)  # sentinel

        worker = await asyncio.to_thread(_produce)
        while True:
            if queue:
                piece = queue.popleft()
                if piece is None:
                    break
                yield piece
            elif error:
                raise _wrap(error[0], self.settings.server_url) from error[0]
            else:
                await asyncio.sleep(0.02)


# --------------------------------------------------------------------------- #
# Tool-call decoding
# --------------------------------------------------------------------------- #

def _decode_tool_calls(raw_calls) -> list[ToolCall]:
    """Decode tool calls from a non-streaming response."""
    calls: list[ToolCall] = []
    for call in raw_calls or []:
        fn = getattr(call, "function", None)
        if fn is None:
            continue
        raw = getattr(fn, "arguments", "") or ""
        calls.append(ToolCall(
            id=str(getattr(call, "id", "") or f"call_{len(calls)}"),
            name=str(getattr(fn, "name", "") or ""),
            arguments=_parse_arguments(raw),
            raw_arguments=raw,
        ))
    return calls


def _accumulate_tool(state: dict[int, dict], tc: Any) -> None:
    """Merge one streamed tool-call delta into the accumulating state."""
    idx = int(getattr(tc, "index", 0) or 0)
    slot = state.setdefault(idx, {"id": "", "name": "", "arguments": ""})
    if getattr(tc, "id", None):
        slot["id"] = str(tc.id)
    fn = getattr(tc, "function", None)
    if fn is not None:
        if getattr(fn, "name", None):
            slot["name"] = str(fn.name)
        if getattr(fn, "arguments", None):
            slot["arguments"] += str(fn.arguments)


def _finalize_tools(state: dict[int, dict]) -> list[ToolCall]:
    """Build ToolCall objects from accumulated streamed deltas."""
    calls: list[ToolCall] = []
    for _, slot in sorted(state.items()):
        if not slot.get("name"):
            continue
        calls.append(ToolCall(
            id=slot.get("id") or f"call_{len(calls)}",
            name=slot["name"],
            arguments=_parse_arguments(slot.get("arguments", "")),
            raw_arguments=slot.get("arguments", ""),
        ))
    return calls


def _parse_arguments(raw: str) -> dict:
    """Parse tool arguments, tolerating the malformed JSON local models produce."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        repaired = _repair_json(raw)
        if repaired is not None:
            return repaired
        log.warning("unparseable tool arguments: %.200s", raw)
        return {"_raw": raw}


def _repair_json(raw: str):
    """Best-effort recovery for truncated or markdown-wrapped JSON.

    Small local models frequently emit ``` fences, trailing commas, or drop a closing
    brace mid-generation. Anything unrecoverable returns None and the caller hands the
    raw text back to the model, which usually re-emits it correctly.
    """
    text = raw.strip()
    if text.startswith("```"):
        fence_count = text.count("```")
        if fence_count > 1:
            text = text.split("```", 2)[-1]
        text = text.lstrip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    candidates = [text]
    for closer in ("}", "]"):
        if not text.endswith(closer):
            candidates.append(text + closer)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Errors and helpers
# --------------------------------------------------------------------------- #

def _safe_chunk(on_chunk: Callable[[str], None], piece: str) -> None:
    """Never let a broken sink take down the request."""
    try:
        on_chunk(piece)
    except Exception as exc:
        log.warning("chunk sink raised: %s", exc)


def _log_completion(completion: Completion, *, extra: dict) -> None:
    fields = {
        "model": completion.model or completion.messages and "unknown",
        "tokens": completion.used_tokens,
        "elapsed_s": round(completion.elapsed_s, 3),
        "tps": round(completion.tokens_per_second, 1),
        "tools": len(completion.tool_calls),
        **extra,
    }
    log.debug(
        "completion %d tok in %.2fs (%.1f tok/s)",
        completion.used_tokens, completion.elapsed_s, completion.tokens_per_second,
        extra=fields,
    )


def _wrap(exc: Exception, server_url: str) -> LLMError:
    """Translate transport errors into an actionable message."""
    message = str(exc)
    lowered = message.lower()
    if "connection" in lowered or "refused" in lowered or "unreachable" in lowered:
        return LLMError(
            f"Cannot reach the LLM server at {server_url}. "
            "Start LM Studio (Developer tab -> Start Server), or pass --server <url>."
        )
    if "timed out" in lowered or "timeout" in lowered:
        return LLMError(f"LLM request timed out at {server_url}")
    if "401" in message or "403" in message or "unauthor" in lowered:
        return LLMError(f"LLM server rejected the request: {message}")
    if "404" in message:
        return LLMError(f"Model not found on {server_url}: {message}")
    return LLMError(message)


# --------------------------------------------------------------------------- #
# Token budgeting
# --------------------------------------------------------------------------- #

def count_tokens(text: str) -> int:
    """A cheap estimate, used before any model-dependent tokenizer is loaded."""
    # ~4 chars/token for English: stable enough for budgeting, exact enough to be safe.
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: Iterable[dict | ChatMessage]) -> int:
    """Estimate the prompt size of a message list, including tool payloads."""
    total = 0
    for message in messages:
        if isinstance(message, ChatMessage):
            message = message.to_dict()
        total += count_tokens(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            total += count_tokens(json.dumps(call, ensure_ascii=False))
    return total
