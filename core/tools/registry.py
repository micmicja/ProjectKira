"""Tool execution: JSON-schema tool definitions and a validated dispatch registry.

A tool is a plain object with a name, a JSON Schema argument definition, a callable,
and an optional approval gate. The registry converts them into the ``tools=[...]``
shape local OpenAI-compatible servers expect, validates the model's arguments against
the schema before calling anything, and runs execution on a worker thread so a slow
tool can never freeze the GUI.

Every failure is captured and returned to the model as a tool result rather than
raised into the agent loop: a tool raising is exactly the case the loop is least able
to recover from, so it must become a message the model can read and respond to.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import threading
import time
import traceback
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from core.log import get_logger, truncate

log = get_logger(__name__)

ToolFn = Callable[..., Any | Awaitable[Any]]


class ToolError(RuntimeError):
    """Raised inside a tool when it needs to report a recoverable failure."""


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #

def _python_type_to_schema(annotation: Any) -> dict:
    """Map a parameter annotation to a minimal JSON Schema type."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    origin = getattr(annotation, "__origin__", None)
    if origin in (list, set, tuple):
        return {"type": "array"}
    if origin in (dict,):
        return {"type": "object"}
    if origin is not None:
        return {"type": "string"}
    return mapping.get(annotation, {"type": "string"})


@dataclass(slots=True)
class Tool:
    """A callable exposed to the model as a function-calling tool."""

    name: str
    description: str
    fn: ToolFn
    parameters: dict = field(default_factory=dict)
    requires_confirmation: bool = False
    timeout_s: float | None = None
    enabled: bool = True
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"tool {self.name!r} needs a description")
        # `fn` and `parameters` are intentionally not cross-checked: a hand-written
        # schema is often a superset of what the callable needs, and letting the
        # model pass through is the usual reason the two diverge.

    # -- schema ---------------------------------------------------------------

    @property
    def schema(self) -> dict:
        """The OpenAI-compatible tool definition sent to the server."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or _default_object_schema(),
            },
        }

    def default_schema(self) -> dict:
        """Generate a JSON Schema from the callable's signature.

        This is a convenience for quick tools, not a replacement for a written schema:
        it cannot express enums, minimums, or "at least one of". Prefer an explicit
        ``parameters=`` for anything the model will be asked to call often.
        """
        sig = inspect.signature(self.fn)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for param in sig.parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            schema = _python_type_to_schema(param.annotation)
            if param.default is inspect.Parameter.empty:
                required.append(param.name)
            else:
                schema["default"] = _json_safe(param.default)
            properties[param.name] = schema
        return {"type": "object", "properties": properties, "required": required}

    # -- invocation -----------------------------------------------------------

    def validate(self, arguments: dict) -> list[str]:
        """Check arguments against the schema; return a list of problems.

        Deliberately shallow: JSON Schema has no official validator bundled, and
        pulling one in for a handful of local tools is not worth the dependency.
        We enforce the things that actually cause model-visible breakage — required
        keys, unknown keys, and type mismatches on declared types.
        """
        problems: list[str] = []
        props = self.parameters.get("properties", {}) if self.parameters else {}
        required = set(self.parameters.get("required", [])) if self.parameters else set()

        missing = sorted(required - set(arguments))
        if missing:
            problems.append(f"missing required argument(s): {', '.join(missing)}")

        for key, value in arguments.items():
            if key not in props:
                continue
            expected = props[key].get("type")
            if expected and not _type_ok(expected, value):
                problems.append(
                    f"argument {key!r} expected {expected}, got {type(value).__name__}"
                )

        return problems

    def invoke(self, arguments: dict) -> Any:
        """Call the tool synchronously with validated arguments."""
        problems = self.validate(arguments)
        if problems:
            raise ToolError("; ".join(problems))
        return self.fn(**arguments)

    async def invoke_async(self, arguments: dict) -> Any:
        """Call the tool, awaiting it when it is a coroutine function."""
        problems = self.validate(arguments)
        if problems:
            raise ToolError("; ".join(problems))
        result = self.fn(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ToolResult:
    """The outcome of one dispatch, ready to be fed back to the model."""

    name: str
    call_id: str
    ok: bool
    output: str
    elapsed_s: float = 0.0
    timed_out: bool = False
    error_type: str | None = None

    @property
    def is_error(self) -> bool:
        return not self.ok

    def as_content(self) -> str:
        """The text the model sees. Errors keep their type so the model can adapt."""
        if self.timed_out:
            return f"[timed out after {self.elapsed_s:.1f}s] {self.output}"
        if self.ok:
            return self.output
        return f"[error: {self.error_type}] {self.output}"

    def to_message_dict(self) -> dict:
        return {
            "role": "tool",
            "content": self.as_content(),
            "tool_call_id": self.call_id,
        }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class Registry:
    """Holds the available tools and dispatches the model's requests."""

    def __init__(self, default_timeout_s: float = 30.0) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()
        self.default_timeout_s = default_timeout_s
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kira-tool")
        self._confirmations: dict[str, threading.Event] = {}
        self._decisions: dict[str, bool] = {}

    # -- registration ---------------------------------------------------------

    def register(self, tool: Tool) -> Tool:
        """Add or replace a tool by name. Returns the tool for chaining."""
        with self._lock:
            self._tools[tool.name] = tool
        log.debug("tool registered: %s", tool.name)
        return tool

    def register_fn(
        self,
        name: str,
        description: str,
        *,
        parameters: dict | None = None,
        requires_confirmation: bool = False,
        timeout_s: float | None = None,
        tags: tuple[str, ...] = (),
    ) -> Callable[[ToolFn], ToolFn]:
        """Decorator form: ``@registry.register_fn("search", "...")``."""

        def decorate(fn: ToolFn) -> ToolFn:
            schema = parameters if parameters is not None else Tool(
                name=name, description=description, fn=fn
            ).default_schema()
            self.register(Tool(
                name=name,
                description=description,
                fn=fn,
                parameters=schema,
                requires_confirmation=requires_confirmation,
                timeout_s=timeout_s,
                tags=tags,
            ))
            return fn

        return decorate

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        with self._lock:
            return self._tools.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    # -- model-facing view ----------------------------------------------------

    def schemas(self, *, tags: list[str] | None = None, enabled_only: bool = True) -> list[dict]:
        """Tool definitions for a chat request, optionally filtered by tag."""
        wanted = set(tags) if tags else None
        with self._lock:
            tools = list(self._tools.values())
        out = []
        for tool in tools:
            if enabled_only and not tool.enabled:
                continue
            if wanted and not (wanted & set(tool.tags)):
                continue
            out.append(tool.schema)
        return sorted(out, key=lambda s: s["function"]["name"])

    def tags(self) -> set[str]:
        with self._lock:
            return {tag for tool in self._tools.values() for tag in tool.tags}

    # -- dispatch -------------------------------------------------------------

    def run(self, name: str, arguments: dict, *, call_id: str = "") -> ToolResult:
        """Run one tool synchronously, capturing every failure into a ToolResult.

        A missing tool, a schema mismatch, a timeout, and a raised exception all
        produce a ToolResult: the agent loop stays alive and the model gets a
        readable error to react to.
        """
        started = time.perf_counter()
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                name=name, call_id=call_id, ok=False,
                output=f"unknown tool {name!r}. Available: {', '.join(self.names()) or 'none'}",
                elapsed_s=0.0, error_type="unknown_tool",
            )
        problems = tool.validate(arguments)
        if problems:
            return ToolResult(
                name=name, call_id=call_id, ok=False,
                output="argument error: " + "; ".join(problems),
                elapsed_s=0.0, error_type="invalid_arguments",
            )

        timeout = tool.timeout_s or self.default_timeout_s
        future = self._executor.submit(tool.invoke, arguments)
        try:
            output = future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            return ToolResult(
                name=name, call_id=call_id, ok=False,
                output=f"{name} did not finish within {timeout:.0f}s",
                elapsed_s=time.perf_counter() - started,
                timed_out=True, error_type="timeout",
            )
        except ToolError as exc:
            return _result(tool.name, call_id, str(exc), started, "tool_error")
        except Exception as exc:
            log.warning("tool %s raised %s: %s", name, type(exc).__name__, exc, exc_info=exc)
            return _result(
                tool.name, call_id, f"{type(exc).__name__}: {exc}", started, type(exc).__name__
            )

        return ToolResult(
            name=name,
            call_id=call_id,
            ok=True,
            output=_render(output),
            elapsed_s=time.perf_counter() - started,
        )

    async def run_async(self, name: str, arguments: dict, *, call_id: str = "") -> ToolResult:
        """Async dispatch. Runs sync tools off the event loop so they cannot stall it."""
        started = time.perf_counter()
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                name=name, call_id=call_id, ok=False,
                output=f"unknown tool {name!r}. Available: {', '.join(self.names()) or 'none'}",
                error_type="unknown_tool",
            )
        problems = tool.validate(arguments)
        if problems:
            return ToolResult(
                name=name, call_id=call_id, ok=False,
                output="argument error: " + "; ".join(problems),
                error_type="invalid_arguments",
            )

        timeout = tool.timeout_s or self.default_timeout_s
        try:
            if tool.requires_confirmation and not self._await_confirmation(name):
                return ToolResult(
                    name=name, call_id=call_id, ok=False,
                    output=f"the user declined to run {name}",
                    elapsed_s=time.perf_counter() - started, error_type="declined",
                )
            output = await asyncio.wait_for(tool.invoke_async(arguments), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return ToolResult(
                name=name, call_id=call_id, ok=False,
                output=f"{name} did not finish within {timeout:.0f}s",
                elapsed_s=time.perf_counter() - started, timed_out=True, error_type="timeout",
            )
        except ToolError as exc:
            return _result(tool.name, call_id, str(exc), started, "tool_error")
        except Exception as exc:
            log.warning("tool %s raised %s: %s", name, type(exc).__name__, exc)
            return _result(tool.name, call_id, f"{type(exc).__name__}: {exc}", started, type(exc).__name__)

        return ToolResult(
            name=name, call_id=call_id, ok=True, output=_render(output),
            elapsed_s=time.perf_counter() - started,
        )

    # -- confirmation gate ----------------------------------------------------

    def confirm(self, name: str, answer: bool) -> None:
        """Record the user's decision for a pending confirmation.

        Called from the GUI (main thread). Decisions are latched before the event
        is set so a race between the two cannot be read as an unanswered prompt.
        """
        with self._lock:
            self._decisions[name] = answer
            event = self._confirmations.get(name)
        if event is not None:
            event.set()

    def request_confirmation(self, name: str, timeout_s: float = 20.0) -> bool:
        """Block until a decision arrives. Used by sync dispatch and tests."""
        event = threading.Event()
        with self._lock:
            self._decisions.pop(name, None)
            self._confirmations[name] = event
        answered = event.wait(timeout_s)
        with self._lock:
            decision = self._decisions.pop(name, None)
            self._confirmations.pop(name, None)
        if not answered:
            log.warning("confirmation for %s timed out", name)
            return False
        return bool(decision)

    async def _await_confirmation(self, name: str) -> bool:
        """Async version of the confirmation gate for ``run_async``."""
        return await asyncio.to_thread(self.request_confirmation, name)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _default_object_schema() -> dict:
    return {"type": "object", "properties": {}}


def _type_ok(expected: str, value: Any) -> bool:
    """Loose JSON Schema type check, matching what the servers actually enforce."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def _render(output: Any) -> str:
    """Turn a tool's return value into model-readable text."""
    if output is None:
        return "[no output]"
    if isinstance(output, str):
        return output
    if isinstance(output, (bytes, bytearray)):
        return f"[binary {len(output)} bytes]"
    if isinstance(output, (dict, list, tuple)):
        try:
            return json.dumps(output, ensure_ascii=False, default=str, indent=2)
        except (TypeError, ValueError):
            return repr(output)
    text = str(output)
    return truncate(text, 4000)


def _result(name: str, call_id: str, output: str, started: float, error_type: str) -> ToolResult:
    return ToolResult(
        name=name, call_id=call_id, ok=False, output=output,
        elapsed_s=time.perf_counter() - started, error_type=error_type,
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def build_registry(tool_factories: Callable[["Registry", Any], None] | None = None) -> Registry:
    """Create a registry and populate it with each factory's tools."""
    registry = Registry()
    for factory in tool_factories or []:
        try:
            factory(registry)
        except Exception as exc:
            log.warning("tool factory %s failed: %s", factory, exc)
    return registry
