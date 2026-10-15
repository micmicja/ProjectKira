"""Built-in tools available to the model out of the box.

Each factory takes a :class:`Registry` and adds its tools. Factories are separate
callables rather than module-level decorators so nothing is registered at import time:
tests and the GUI can build a subset, and a broken optional dependency degrades one
factory instead of the whole import.

Every tool here is read-only. Anything that writes to disk, the network, or the system
should live in a plugin and declare ``requires_confirmation=True``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from core.tools.registry import Registry, Tool


# --------------------------------------------------------------------------- #
# Time and date
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def register_time_tools(registry: Registry) -> None:
    """Clock, calendar, and conversion tools. No dependencies beyond stdlib."""

    def current_time(timezone: str = "") -> dict:
        """Return the current date and time.

        Args:
            timezone: An IANA zone name such as "Europe/Berlin". Empty for the
                system local time.
        """
        zone = _safe_zone(timezone)
        now = dt.datetime.now(zone)
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%A, %B %d, %Y"),
            "iso": now.isoformat(timespec="seconds"),
            "timezone": str(zone),
        }

    def convert_time(
        hour: float,
        minute: int = 0,
        from_timezone: str = "",
        to_timezone: str = "",
    ) -> dict:
        """Convert a time of day between timezones.

        Args:
            hour: Hour of day, 0-23, may be fractional.
            minute: Minutes past the hour.
            from_timezone: IANA source zone, empty for system local.
            to_timezone: IANA destination zone, empty for system local.
        """
        source = _safe_zone(from_timezone)
        target = _safe_zone(to_timezone)
        now = dt.datetime.now(source)
        moment = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
        converted = moment.astimezone(target)
        return {
            "from": str(source),
            "to": str(target),
            "from_time": moment.strftime("%H:%M"),
            "to_time": converted.strftime("%H:%M"),
            "to_date": converted.strftime("%A, %B %d, %Y"),
            "offset_hours": round(converted.utcoffset().total_seconds() / 3600, 1),
        }

    def days_until(month: int, day: int, year: int | None = None) -> dict:
        """Count days from today to a date.

        Args:
            month: 1-12.
            day: 1-31.
            year: Year of the target. Empty/None means this year, or next year if
                this year's date has already passed.
        """
        today = dt.date.today()
        target_year = year if year else today.year
        try:
            target = dt.date(target_year, month, day)
        except ValueError:
            return {"error": f"no such date: {target_year}-{month:02d}-{day:02d}"}
        if year is None and target < today:
            target = dt.date(target_year + 1, month, day)
        delta = (target - today).days
        weekday = target.strftime("%A")
        return {"date": target.isoformat(), "weekday": weekday, "days": delta}

    def day_of_week(month: int, day: int, year: int | None = None) -> dict:
        """Return the weekday for a date.

        Args:
            month: 1-12.
            day: 1-31.
            year: Year; None means the current year.
        """
        year = year or dt.date.today().year
        try:
            date = dt.date(year, month, day)
        except ValueError:
            return {"error": f"no such date: {year}-{month:02d}-{day:02d}"}
        return {"date": date.isoformat(), "weekday": date.strftime("%A")}

    registry.register(Tool(
        name="current_time",
        description="Get the current local date and time, optionally in another timezone.",
        fn=current_time,
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": 'IANA timezone name, e.g. "Europe/Berlin". Empty for local time.',
                },
            },
        },
        tags=("time",),
    ))

    registry.register(Tool(
        name="convert_time",
        description="Convert a time of day between two timezones.",
        fn=convert_time,
        parameters={
            "type": "object",
            "properties": {
                "hour": {"type": "number", "description": "Hour of day, 0-23, may be fractional."},
                "minute": {"type": "integer", "description": "Minutes, 0-59.", "default": 0},
                "from_timezone": {"type": "string", "description": "IANA source zone, empty for local."},
                "to_timezone": {"type": "string", "description": "IANA destination zone, empty for local."},
            },
            "required": ["hour"],
        },
        tags=("time",),
    ))

    registry.register(Tool(
        name="days_until",
        description="How many days from today until a given month and day.",
        fn=days_until,
        parameters={
            "type": "object",
            "properties": {
                "month": {"type": "integer", "minimum": 1, "maximum": 12},
                "day": {"type": "integer", "minimum": 1, "maximum": 31},
                "year": {"type": "integer", "description": "Optional; defaults to this year."},
            },
            "required": ["month", "day"],
        },
        tags=("time",),
    ))

    registry.register(Tool(
        name="day_of_week",
        description="Which weekday a given date falls on.",
        fn=day_of_week,
        parameters={
            "type": "object",
            "properties": {
                "month": {"type": "integer", "minimum": 1, "maximum": 12},
                "day": {"type": "integer", "minimum": 1, "maximum": 31},
                "year": {"type": "integer"},
            },
            "required": ["month", "day"],
        },
        tags=("time",),
    ))


def _safe_zone(name: str):
    """Parse an IANA zone name, falling back to the system local zone."""
    name = (name or "").strip()
    if not name:
        return dt.datetime.now().astimezone().tzinfo
    try:
        return ZoneInfo(name)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #

def _solve(expression: str) -> dict:
    """Evaluate an arithmetic expression in a restricted, name-free sandbox."""
    if not expression or not expression.strip():
        raise ValueError("no expression given")

    allowed = set("0123456789+-*/().,%eE ") | {"pi", "tau"}
    if any(ch not in allowed for ch in expression):
        raise ValueError(
            f"expression contains characters that are not allowed: {expression!r}"
        )
    constants = {
        "pi": math.pi,
        "tau": math.tau,
        "__builtins__": {},
        "pow": pow,
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
    }
    try:
        value = eval(expression, {"__builtins__": {}}, constants)  # noqa: S307
    except ZeroDivisionError:
        return {"error": "division by zero", "expression": expression}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "expression": expression}
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return {"expression": expression, "result": value}


def register_math_tools(registry: Registry) -> None:
    """A sandboxed arithmetic evaluator and unit-less helpers."""

    registry.register(Tool(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression. Supports + - * / ** % (), decimals, "
            "pi and tau, and the functions sqrt sin cos tan log log10 exp pow abs min max. "
            "Use this for any calculation you would otherwise do in your head."
        ),
        fn=lambda expression: _solve(expression),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": 'e.g. "12 * (34 + 8)" or "sqrt(2) * 10"',
                },
            },
            "required": ["expression"],
        },
        tags=("math",),
    ))


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def register_memory_tools(registry: Registry, store_path: Path) -> None:
    """A tiny on-disk key/value memory the assistant can actually recall.

    Deliberately flat and file-backed: it survives restarts, it is trivially
    inspectable, and it cannot do anything except read and write its own file.
    Writes are confirmation-gated so the model cannot silently accumulate state.
    """
    store_path = Path(store_path)

    def _load() -> dict:
        if not store_path.exists():
            return {}
        try:
            data = json.loads(store_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(data: dict) -> None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def remember(key: str, value: str) -> dict:
        """Store a fact to recall later.

        Args:
            key: Short label for the fact, e.g. "office_hours".
            value: The fact itself, in plain prose.
        """
        key = key.strip().lower().replace(" ", "_")
        if not key or not value.strip():
            return {"error": "both key and value are required"}
        data = _load()
        data[key] = {"value": value.strip(), "saved": _now_iso()}
        _save(data)
        return {"stored": key, "count": len(data)}

    def recall(key: str) -> dict:
        """Look up a previously stored fact.

        Args:
            key: The label that was stored with remember.
        """
        data = _load()
        hit = data.get(key.strip().lower().replace(" ", "_"))
        if hit is None:
            return {"found": False, "key": key, "available": sorted(data)[:50]}
        return {"found": True, "key": key, "value": hit["value"], "saved": hit.get("saved", "")}

    def list_memory() -> dict:
        """List every stored fact, most recent last."""
        data = _load()
        entries = sorted(data.items(), key=lambda kv: kv[1].get("saved", ""))
        return {
            "count": len(entries),
            "entries": [{"key": k, "value": v.get("value", "")} for k, v in entries],
        }

    def forget(key: str) -> dict:
        """Delete a stored fact.

        Args:
            key: The label to remove.
        """
        data = _load()
        norm = key.strip().lower().replace(" ", "_")
        if norm not in data:
            return {"removed": False, "key": key}
        del data[norm]
        _save(data)
        return {"removed": True, "key": key, "count": len(data)}

    registry.register(Tool(
        name="remember",
        description="Store a fact so it can be recalled across sessions.",
        fn=remember,
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short label, e.g. 'office_hours'."},
                "value": {"type": "string", "description": "The fact, in plain prose."},
            },
            "required": ["key", "value"],
        },
        requires_confirmation=True,
        tags=("memory", "write"),
    ))

    registry.register(Tool(
        name="recall",
        description="Look up a fact previously stored with remember.",
        fn=recall,
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        tags=("memory",),
    ))

    registry.register(Tool(
        name="list_memory",
        description="List every fact currently stored in memory.",
        fn=list_memory,
        parameters={"type": "object", "properties": {}},
        tags=("memory",),
    ))

    registry.register(Tool(
        name="forget",
        description="Delete a stored fact from memory.",
        fn=forget,
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        requires_confirmation=True,
        tags=("memory", "write"),
    ))


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #

def register_system_tools(registry: Registry) -> None:
    """Host diagnostics. Read-only; useful for 'how much memory is left?'."""

    def system_info() -> dict:
        """Return host, platform, and Python information."""
        load = None
        try:
            load = os.getloadavg()
        except (OSError, AttributeError):
            pass
        return {
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "load_avg": load,
            "time": _now_iso(),
        }

    def disk_usage(path: str = ".") -> dict:
        """Return free and total disk space.

        Args:
            path: Path to measure. Defaults to the current directory.
        """
        target = Path(path).expanduser()
        usage = shutil_disk_usage(target)
        return {
            "path": str(target.resolve()),
            "total_gb": round(usage.total / 1024 ** 3, 1),
            "used_gb": round(usage.used / 1024 ** 3, 1),
            "free_gb": round(usage.free / 1024 ** 3, 1),
            "percent_used": round(usage.used / usage.total * 100, 1) if usage.total else 0,
        }

    registry.register(Tool(
        name="system_info",
        description="Get host, OS, and Python runtime information.",
        fn=system_info,
        parameters={"type": "object", "properties": {}},
        tags=("system",),
    ))

    registry.register(Tool(
        name="disk_usage",
        description="Get free and used disk space at a path.",
        fn=disk_usage,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to measure."}},
        },
        tags=("system",),
    ))


def shutil_disk_usage(path: Path):
    import shutil as _sh

    return _sh.disk_usage(path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def register_builtins(registry: Registry, *, data_dir: Path | None = None) -> Registry:
    """Register every built-in tool set that is safe to enable by default."""
    register_time_tools(registry)
    register_math_tools(registry)
    register_system_tools(registry)
    if data_dir is not None:
        register_memory_tools(registry, Path(data_dir) / "memory.json")
    return registry


def default_registry(data_dir: Path | None = None) -> Registry:
    """A ready-to-use registry with the standard built-ins."""
    return register_builtins(Registry(), data_dir=data_dir)
