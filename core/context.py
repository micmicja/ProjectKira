"""Conversation state: identity prompt, capability summary, and sliding-window history.

Three separate concerns, because each is testable without the others:

* :func:`build_system_prompt` -- the assistant's identity, speech rules, and the tools
  available this turn. Rebuilt per request rather than cached, since the tool set
  changes as users enable and disable things.
* :class:`SlidingWindow` -- holds the transcript and keeps it under a token budget.
  This is the VRAM protection: an unbounded voice conversation would otherwise grow
  every prompt until the local model stalls.
* :class:`ContextManager` -- the two together, plus the one thing callers mostly want:
  the exact message list to send to the model right now.

The window keeps a rolling summary of anything it evicts, so a long conversation loses
detail gradually instead of going quiet about earlier turns.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from core.llm import ChatMessage, count_tokens, estimate_messages_tokens
from core.log import get_logger

log = get_logger(__name__)


#: Hard cap on the identity prompt. It is re-sent with every request, so each token
#: here is paid for on the first prompt-cache miss and again on every miss after.
MAX_SYSTEM_PROMPT_TOKENS = 900


# Speech rules live here rather than in settings: they describe the output medium,
# which is fixed, not a per-user preference.
_SPEECH_RULES = (
    "Speak like a person, not a document: short sentences, no markdown, no bullet "
    "lists, no tables, no code blocks. Never read file paths or JSON aloud verbatim; "
    "summarise results in a sentence or two."
)


def build_system_prompt(
    *,
    name: str = "KIRA",
    personality: str = "",
    tool_schemas: list[dict] | None = None,
    extra_instructions: str | None = None,
    max_tokens: int = MAX_SYSTEM_PROMPT_TOKENS,
) -> str:
    """Assemble the system prompt: identity, speech rules, and available tools.

    Tool schemas are folded in as a compact capability list rather than pasted as raw
    JSON. The server already receives full schemas via ``tools=[...]``, so repeating
    them here would double prompt cost for no benefit. The model needs a name and a
    one-line description to decide whether to call something; it does not need the
    parameter grammar in the prose.
    """
    sections: list[str] = []

    identity = f"You are {name}, a local voice assistant."
    if personality:
        identity += " " + personality.strip()
    identity += (
        " Everything you say is spoken aloud immediately by a text-to-speech engine, "
        "and the user is listening while it plays."
    )
    sections.append(identity)
    sections.append(_SPEECH_RULES)

    if tool_schemas:
        lines = [
            f"- {schema['function']['name']}: {schema['function'].get('description', '').strip()}"
            for schema in tool_schemas
        ]
        sections.append(
            "Tools you can call. Call one only when it directly answers the request -- "
            "not to sound thorough. If a tool returns an error, say so plainly and offer "
            "a simpler path.\n\n" + "\n".join(lines)
        )

    sections.append(
        "If you are unsure, say so briefly. Never invent facts, numbers, or results "
        "that a tool did not return to you."
    )
    if extra_instructions:
        sections.append(extra_instructions.strip())

    prompt = "\n\n".join(section for section in sections if section)
    budget_chars = max_tokens * 4
    if len(prompt) > budget_chars:
        prompt = prompt[:budget_chars].rsplit(" ", 1)[0]
        log.warning("system prompt truncated to %d chars", len(prompt))
    return prompt


# --------------------------------------------------------------------------- #
# Sliding window
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class WindowStats:
    """A snapshot of window state, for logging and the HUD."""

    messages: int
    estimated_tokens: int
    budget_tokens: int
    summary_tokens: int
    evicted: int
    last_summary_at: float | None


#: Anything that can compress a list of messages into a short digest. Returning
#: ``None`` means "I could not", and the window falls back to hard trimming.
Summariser = Callable[[list[ChatMessage], str], "str | None"]


class SlidingWindow:
    """Holds the transcript and keeps it inside a token budget.

    The system prompt is deliberately not counted here: it is rebuilt per request and
    is constant in size, so folding it in would make eviction behaviour depend on how
    verbose the tool descriptions are. The budget applies only to history plus the
    rolling summary, which keeps turn counts predictable.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 4096,
        keep_recent: int = 12,
        summariser: Summariser | None = None,
    ) -> None:
        if max_tokens < 256:
            raise ValueError("max_tokens must be at least 256")
        if keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
        self.max_tokens = int(max_tokens)
        self.keep_recent = int(keep_recent)
        self._summariser = summariser
        self._messages: list[ChatMessage] = []
        self._summary = ""
        self._evicted = 0
        self._last_summary_at: float | None = None

    # -- mutation -----------------------------------------------------------

    def append(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def extend(self, messages: list[ChatMessage]) -> None:
        self._messages.extend(messages)

    def clear(self) -> None:
        self._messages = []
        self._summary = ""
        self._evicted = 0
        self._last_summary_at = None

    # -- reads --------------------------------------------------------------

    def messages(self) -> list[ChatMessage]:
        """A copy, so callers cannot mutate history underneath the window."""
        return list(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def evicted(self) -> int:
        return self._evicted

    @property
    def last_summary_at(self) -> float | None:
        return self._last_summary_at

    # -- accounting ---------------------------------------------------------

    def _history_tokens(self) -> int:
        return estimate_messages_tokens(m.to_dict() for m in self._messages)

    def estimated_tokens(self) -> int:
        """Conversation cost including the rolling summary."""
        total = self._history_tokens()
        return total + count_tokens(self._summary) if self._summary else total

    def stats(self) -> WindowStats:
        return WindowStats(
            messages=len(self._messages),
            estimated_tokens=self.estimated_tokens(),
            budget_tokens=self.max_tokens,
            summary_tokens=count_tokens(self._summary) if self._summary else 0,
            evicted=self._evicted,
            last_summary_at=self._last_summary_at,
        )

    # -- compaction ---------------------------------------------------------

    def fit(self, headroom_tokens: int = 0) -> bool:
        """Compact until history fits in ``max_tokens - headroom_tokens``.

        ``headroom_tokens`` is what the caller intends to add next, typically the user
        message being appended. Charging it ahead of time means the request cannot
        overshoot the model's context window the instant it is sent.

        Returns True when the window changed.
        """
        target = max(256, self.max_tokens - int(headroom_tokens))
        before = (len(self._messages), self._summary)
        while self._history_tokens() > target:
            if not self.compact():
                break
        return (len(self._messages), self._summary) != before

    def compact(self) -> bool:
        """Shrink history to fit the budget. Returns True if anything changed.

        Strategy: keep the newest ``keep_recent`` messages, summarise everything older,
        and re-insert that digest as a system message. When summarisation is unavailable
        the oldest half is dropped instead -- lossier, but still correct, and it never
        fails the user's turn.

        The cut point walks backward past ``tool`` messages and past the assistant turn
        that owns its tool_calls. Splitting inside a call/result pair yields a request
        most local servers reject outright, which would turn a slow compaction into a
        hard error on the next turn.
        """
        if self._history_tokens() <= self.max_tokens:
            return False

        cut = _safe_cut_point(self._messages, self.keep_recent)
        if cut <= 0:
            # Everything old would be lost either way; keep the newest messages and
            # drop the tail of the oldest run instead of compaction failing.
            return self._hard_trim()

        old, recent = self._messages[:cut], self._messages[cut:]
        digest = self._summarise(old)
        if digest is None:
            return self._hard_trim()

        self._summary = digest
        self._messages = [_summary_message(digest)] + recent
        self._last_summary_at = time.time()
        log.info(
            "compacted %d messages into a %d-token summary",
            len(old), count_tokens(digest),
        )
        return True

    def _summarise(self, old: list[ChatMessage]) -> str | None:
        """Ask the summariser to compress ``old``. Returns None on any failure."""
        if self._summariser is None or not old:
            return None
        try:
            result = self._summariser(old, self._summary)
        except Exception as exc:
            log.warning("summariser failed: %s", exc)
            return None
        if not result or not result.strip():
            return None
        return result.strip()

    def _hard_trim(self) -> bool:
        """Drop the oldest messages until history fits, keeping call/result pairs intact.

        Returns True if any messages were removed *this call*.
        """
        evicted_this_call = 0
        while self._history_tokens() > self.max_tokens and len(self._messages) > self.keep_recent:
            # Find a safe drop point: walk back from the front past any tool results
            # and the assistant turn that owns them.
            drop_end = 1
            while drop_end < len(self._messages):
                msg = self._messages[drop_end]
                if msg.role == "tool":
                    drop_end += 1
                    continue
                if msg.role == "assistant" and msg.tool_calls:
                    drop_end += 1
                    continue
                break

            if drop_end == 0:
                # No safe place to cut without breaking call/result pairs.
                break

            removed = self._messages[:drop_end]
            del self._messages[:drop_end]
            evicted_this_call += len(removed)

        if evicted_this_call:
            self._evicted += evicted_this_call
            log.warning("hard-trimmed %d messages without summarisation", evicted_this_call)
        return evicted_this_call > 0


# --------------------------------------------------------------------------- #
# Cut-point helpers
# --------------------------------------------------------------------------- #

def _safe_cut_point(messages: list[ChatMessage], keep_recent: int) -> int:
    """Index at which history may be split, never between a call and its result.

    A ``tool`` message must not sit at the cut: it would be summarised away while its
    assistant turn stays visible, or vice versa. Walk back past the whole pair.
    """
    cut = max(0, len(messages) - keep_recent)
    while 0 < cut < len(messages):
        message = messages[cut]
        if message.role == "tool":
            cut -= 1
            continue
        if message.role == "assistant" and message.tool_calls:
            # Keep this turn with the results it produced.
            cut -= 1
            continue
        return cut
    return 0


def _summary_message(text: str) -> ChatMessage:
    """The system message that carries the rolling digest back into the prompt."""
    body = (
        "Earlier in this conversation the following was established. Use it as "
        "background and do not repeat it back unless asked.\n\n" + text.strip()
    )
    return ChatMessage(role="system", content=body)


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ContextManager:
    """Owns the identity prompt and the sliding window for one assistant."""

    assistant_name: str = "KIRA"
    personality: str = "You are concise, warm, and practical."
    max_context_tokens: int = 4096
    keep_recent: int = 12
    max_system_prompt_tokens: int = MAX_SYSTEM_PROMPT_TOKENS
    extra_instructions: str | None = None
    summariser: Summariser | None = None

    window: SlidingWindow = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.window = SlidingWindow(
            max_tokens=self.max_context_tokens,
            keep_recent=self.keep_recent,
            summariser=self.summariser,
        )

    # -- prompt -------------------------------------------------------------

    def system_prompt(self, tool_schemas: list[dict] | None = None) -> str:
        """The system prompt for this turn, reflecting the live tool set."""
        return build_system_prompt(
            name=self.assistant_name,
            personality=self.personality,
            tool_schemas=tool_schemas,
            extra_instructions=self.extra_instructions,
            max_tokens=self.max_system_prompt_tokens,
        )

    def system_message(self, tool_schemas: list[dict] | None = None) -> ChatMessage:
        return ChatMessage(role="system", content=self.system_prompt(tool_schemas))

    # -- conversation -------------------------------------------------------

    def add_user(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # Reserve the cost of this message before adding it, so a single append cannot
        # push the request past the model's window.
        self.window.fit(headroom_tokens=count_tokens(text) + 32)
        self.window.append(ChatMessage(role="user", content=text))

    def add_assistant(
        self,
        content: str,
        *,
        tool_calls: list[dict] | None = None,
    ) -> None:
        self.window.append(
            ChatMessage(role="assistant", content=content or "", tool_calls=tool_calls or [])
        )

    def add_tool_result(
        self, call_id: str, content: str, *, name: str | None = None
    ) -> None:
        self.window.append(
            ChatMessage(role="tool", content=content, tool_call_id=call_id, name=name)
        )

    def add_messages(self, messages: list[ChatMessage]) -> None:
        self.window.extend(messages)

    def compact(self) -> bool:
        """Force compaction regardless of budget. Useful in tests and on "reset"."""
        return self.window.compact()

    def reset(self) -> None:
        self.window.clear()

    # -- request building ----------------------------------------------------

    def build_messages(self, *, tool_schemas: list[dict] | None = None) -> list[ChatMessage]:
        """The exact message list to send to the model right now.

        The system prompt is rebuilt and never stored in the window, so identity and
        the tool list reflect this turn rather than the conversation's first request.
        """
        return [self.system_message(tool_schemas)] + self.window.messages()

    def stats(self) -> WindowStats:
        return self.window.stats()

    def estimate_turn_cost(self, extra_text: str = "") -> int:
        """Tokens the next request will need, before it is built."""
        cost = self.window.estimated_tokens() + count_tokens(self.system_prompt())
        return cost + (count_tokens(extra_text) if extra_text else 0)

    # -- persistence ---------------------------------------------------------

    def to_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self.window.messages()]

    def load(self, messages: list[dict]) -> None:
        self.window.clear()
        self.window.extend([ChatMessage.from_dict(m) for m in messages])


def render_tool_summary(tool_schemas: list[dict]) -> str:
    """A compact text rendering of the tool set, for logs and the HUD."""
    if not tool_schemas:
        return "(no tools available)"
    return "\n".join(
        f"  {schema['function']['name']} — {schema['function'].get('description', '').strip()}"
        for schema in tool_schemas
    )


__all__ = [
    "MAX_SYSTEM_PROMPT_TOKENS",
    "ContextManager",
    "SlidingWindow",
    "Summariser",
    "WindowStats",
    "build_system_prompt",
    "render_tool_summary",
]
