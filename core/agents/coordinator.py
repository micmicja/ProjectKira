"""Conversation and tool orchestration.

One coordinator owns a conversation's state and the agent loop that follows a user
turn. Keeping that state in one place is what makes the rest of the system simple:
audio, GUI, and text input are all just callers of :meth:`chat`.

    coordinator.chat("what time is it?")
        -> sends a completion, runs any tool calls it asked for, and returns a final
           spoken answer once per turn.

The loop runs to a configurable depth so a tool result can trigger one more model call
to summarise it — but never indefinitely.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings
from core.llm import (
    ChatMessage,
    Completion,
    LLMClient,
    ToolCall,
    count_tokens,
    estimate_messages_tokens,
)
from core.log import get_logger, truncate
from core.tools.registry import Registry, ToolResult


log = get_logger(__name__)


#: Event sink for the GUI. Callers adapt to ``(event=, text=, **extra)`` and are
#: wrapped so a broken UI hook can never take down a turn.
OnProgress = Callable[..., None]


@dataclass(slots=True)
class ToolEvent:
    """One tool execution, for the transcript and HUD."""

    name: str
    call_id: str
    arguments: dict
    ok: bool
    output: str
    elapsed_s: float


@dataclass(slots=True)
class Turn:
    """Everything that happened on a single user request."""

    user: str
    answer: str
    elapsed_s: float
    tool_events: list[ToolEvent] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def used_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ConversationCoordinator:
    """Owns conversation history, context trimming, and the tool-calling loop."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        registry: Registry,
        settings: Settings | None = None,
        on_progress: OnProgress | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or Settings()
        self.agent = self.settings.agent
        self._messages: list[ChatMessage] = []
        self._summary: str = ""
        self._lock = threading.RLock()
        self.on_progress = on_progress

        system_prompt = self.agent.system_prompt
        if system_prompt:
            self._messages.append(ChatMessage(role="system", content=system_prompt))

    # -- state ----------------------------------------------------------------

    @property
    def history(self) -> list[ChatMessage]:
        """A copy, so callers cannot mutate history underneath us."""
        with self._lock:
            return list(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    def reset(self) -> None:
        """Drop the conversation, keeping the system prompt."""
        with self._lock:
            self._messages = [m for m in self._messages if m.role == "system"]
            self._summary = ""
        log.info("conversation reset")

    def _estimate_tokens(self) -> int:
        """Rough context size, including the rolling summary."""
        estimate = estimate_messages_tokens(m.to_dict() for m in self._messages)
        if self._summary:
            estimate += count_tokens(self._summary)
        return estimate

    # -- context management ---------------------------------------------------

    def _trim(self) -> None:
        """Keep recent messages plus a rolling summary of everything before.

        Summarization is best-effort: if the model call fails we fall back to hard
        truncation, which is worse but still correct, and log it rather than failing
        the user's turn. The split point is always moved back to a message boundary
        that is safe to cut, so a ``tool`` result is never orphaned from its call.
        """
        if self._estimate_tokens() <= self.agent.summary_trigger_tokens:
            return

        kept = self.agent.keep_recent_messages
        cut = max(1, len(self._messages) - kept)
        # Never cut between a tool call and its result.
        while cut < len(self._messages) - 1 and self._messages[cut].role == "tool":
            cut -= 1

        old = self._messages[1:cut] if cut > 1 else []
        recent = self._messages[cut:]
        if not old:
            return  # Nothing to compress without destroying the system prompt.

        summary_text = self._summarize(old, self._summary)
        if summary_text is None:
            drop = max(1, len(old) // 2)
            self._messages = [self._messages[0]] + old[drop:] + recent
            log.warning("summarization unavailable; dropped %d old messages", drop)
            return

        self._summary = summary_text
        self._messages = [
            self._messages[0],
            ChatMessage(role="system", content=_summary_block(summary_text)),
        ] + recent
        log.info(
            "compacted %d messages into a summary", len(old),
            extra={"estimated_tokens": self._estimate_tokens()},
        )

    def _summarize(self, messages: list[ChatMessage], previous: str) -> str | None:
        """Ask the model to compress old messages. Returns None on any failure."""
        if not messages:
            return previous or ""

        instruction = (
            "Summarise this conversation in under 400 words, second person, covering "
            "decisions made, facts stated, and pending tasks. Drop pleasantries. "
            "Return plain text only."
        )
        if previous:
            instruction = (
                "Extend this existing summary with the new material.\n\n"
                f"<previous>\n{previous}\n</previous>\n\n{instruction}"
            )
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages if m.content)

        try:
            result = self.llm.chat(
                [ChatMessage(role="user", content=f"{instruction}\n\n<transcript>\n{transcript}\n</transcript>")],
                max_tokens=600,
                temperature=0.0,
            )
            return result.text.strip() or previous
        except Exception as exc:
            log.warning("summarization failed: %s", exc)
            return None

    # -- main entry point -----------------------------------------------------

    def chat(self, user_text: str, *, tools: list[str] | None = None) -> Turn:
        """Handle one user message end to end and return the final spoken answer."""
        user_text = (user_text or "").strip()
        started = time.perf_counter()

        if not user_text:
            return Turn(user="", answer="", elapsed_s=0.0)

        self._emit("user", user_text)
        with self._lock:
            self._messages.append(ChatMessage(role="user", content=user_text))
            self._trim()
            working = list(self._messages)

        events: list[ToolEvent] = []
        prompt_tokens = 0
        completion_tokens = 0
        tool_schemas = self.registry.schemas(tags=tools, enabled_only=True)

        completion: Completion | None = None
        for depth in range(self.agent.max_turns_per_request):
            completion = self.llm.chat(
                working,
                tools=tool_schemas or None,
                tool_choice="auto" if tool_schemas else "auto",
            )
            prompt_tokens = max(prompt_tokens, completion.prompt_tokens)
            completion_tokens += completion.completion_tokens
            self._emit("assistant", completion.text, extra={"depth": depth})

            if not completion.tool_calls:
                answer = completion.text.strip()
                with self._lock:
                    self._messages.append(ChatMessage(role="assistant", content=answer))
                self._emit("done", answer)
                return Turn(
                    user=user_text, answer=answer,
                    elapsed_s=time.perf_counter() - started,
                    tool_events=events,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                )

            with self._lock:
                working.append(ChatMessage(
                    role="assistant",
                    content=completion.text,
                    tool_calls=[c.to_dict() for c in completion.tool_calls],
                ))
            for call in completion.tool_calls:
                event = self._execute_tool(call)
                events.append(event)
                with self._lock:
                    working.append(call.result_message(event.output))

        # Ran out of depth: return whatever the model last produced.
        answer = completion.text.strip() if completion is not None else ""
        if answer:
            with self._lock:
                self._messages.append(ChatMessage(role="assistant", content=answer))
        self._emit("done", answer)
        return Turn(
            user=user_text, answer=answer,
            elapsed_s=time.perf_counter() - started,
            tool_events=events, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    def chat_stream(
        self,
        user_text: str,
        *,
        on_chunk: Callable[[str], None] | None = None,
        tools: list[str] | None = None,
    ) -> Turn:
        """Same as :meth:`chat`, but forwards tokens as they arrive.

        Streaming is what lets TTS start speaking before the model finishes. The sink
        receives raw chunks, not sentences: the TTS layer owns sentence splitting.

        When tools are registered the request falls back to the non-streaming path — a
        partial tool call gives the speaker text with no way to cancel it.
        """
        user_text = (user_text or "").strip()
        started = time.perf_counter()
        if not user_text:
            return Turn(user="", answer="", elapsed_s=0.0)

        self._emit("user", user_text)
        with self._lock:
            self._messages.append(ChatMessage(role="user", content=user_text))
            self._trim()
            working = list(self._messages)

        tool_schemas = self.registry.schemas(tags=tools, enabled_only=True)
        if not tool_schemas:
            chunks: list[str] = []

            def sink(piece: str) -> None:
                chunks.append(piece)
                if on_chunk is not None:
                    on_chunk(piece)

            completion = self.llm.stream_chat(working, on_chunk=sink)
            answer = completion.text.strip()
            with self._lock:
                self._messages.append(ChatMessage(role="assistant", content=answer))
            self._emit("done", answer)
            return Turn(
                user=user_text, answer=answer,
                elapsed_s=time.perf_counter() - started,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )

        return self.chat(user_text, tools=tools)

    def _execute_tool(self, call: ToolCall) -> ToolEvent:
        """Dispatch one tool call, capturing every failure into a ToolEvent."""
        started = time.perf_counter()
        self._emit("tool_start", call.name, extra={"arguments": call.arguments})

        tool = self.registry.get(call.name)
        if tool is not None and tool.requires_confirmation:
            # Confirmations need a UI attached. A headless run proceeds rather than
            # hanging, and says so in the log.
            if self.on_progress is None:
                log.info("tool %s needs confirmation; no UI attached, proceeding", call.name)

        try:
            result = self.registry.run(call.name, call.arguments, call_id=call.id)
        except Exception as exc:
            result = ToolResult(
                name=call.name, call_id=call.id, ok=False,
                output=f"dispatch error: {exc}", error_type=type(exc).__name__,
            )

        elapsed = time.perf_counter() - started
        output = truncate(result.as_content(), self.agent.max_tool_result_chars)
        self._emit(
            "tool_end", output,
            extra={"name": call.name, "ok": result.ok, "ms": round(elapsed * 1000, 1)},
        )
        return ToolEvent(
            name=call.name, call_id=call.id, arguments=call.arguments,
            ok=result.ok, output=output, elapsed_s=elapsed,
        )

    def _emit(self, event: str, text: str, *, extra: dict[str, Any] | None = None) -> None:
        """Notify the progress sink. Never raises: a broken UI hook must not kill a turn."""
        if self.on_progress is None:
            return
        try:
            self.on_progress(event=event, text=text, **(extra or {}))
        except Exception as exc:
            log.debug("progress sink error: %s", exc)

    # -- serialization --------------------------------------------------------

    def to_dicts(self) -> list[dict]:
        with self._lock:
            return [m.to_dict() for m in self._messages]

    def load(self, messages: list[dict]) -> None:
        with self._lock:
            self._messages = [ChatMessage.from_dict(m) for m in messages]


def _summary_block(summary: str) -> str:
    """The system message that carries the rolling summary back into the prompt."""
    return (
        "Earlier in this conversation, the following was established. Use it as "
        "background; do not repeat it back unless asked.\n\n" + summary.strip()
    )
