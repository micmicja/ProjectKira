"""Structured logging with per-stage latency measurement.

Console output is human-oriented; the file gets JSON lines so a session can be
replayed and correlated after the fact.

    setup_logging(level="INFO", log_dir=Path("data/logs"))
    log = get_logger("core.audio")
    log.info("device changed", extra={"device": 3})

    with log.timed("stt"):
        text = transcribe(wav)
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

_CONSOLE_FMT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s | %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields merged in, never raised."""

    _BASE = {
        "args",
        "asctime",
        "exc_info",
        "exc_text",
        "extra",
        "facility",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msg",
        "message",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        data: dict = {
            "t": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self._BASE or key.startswith("_"):
                continue
            data[key] = _json_safe(value)
        return json.dumps(data, ensure_ascii=False, default=str)


def _json_safe(value):
    """Coerce non-JSON values (timedelta, Path, bytes) to something serializable."""
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


class TimingScope:
    """``with log.timed("stt"): ...`` — measure and report a stage."""

    __slots__ = ("_logger", "stage", "ms", "_extra", "_start")

    def __init__(self, logger: logging.Logger, stage: str, extra: dict | None) -> None:
        self._logger = logger
        self.stage = stage
        self.ms = 0.0
        self._extra = dict(extra or {})
        self._start = 0.0

    def __enter__(self) -> "TimingScope":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ms = (time.perf_counter() - self._start) * 1000.0
        fields = dict(self._extra)
        fields["stage"] = self.stage
        fields["ms"] = round(self.ms, 1)
        if exc_type is not None:
            self._logger.error(
                f"{self.stage} failed in {self.ms:.0f} ms: {exc}", extra=fields
            )
        else:
            self._logger.info(f"{self.stage} {self.ms:.0f} ms", extra=fields)
        return False  # Never swallow the exception.


class LogAdapter(logging.LoggerAdapter):
    """A logger with a :meth:`timed` helper. Use ``get_logger(__name__)``."""

    def timed(self, stage: str, **extra) -> TimingScope:
        return TimingScope(self.logger, stage, extra or None)


def get_logger(name: str) -> LogAdapter:
    logger = logging.getLogger(name)
    logger.setLevel(logging.NOTSET)  # The root logger decides.
    return LogAdapter(logger, {})


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

_NOISY = (
    "urllib3",
    "httpx",
    "httpcore",
    "requests",
    "huggingface_hub",
    "transformers",
    "faster_whisper",
    "tokenizers",
    "filelock",
    "PIL",
    "matplotlib",
)


def setup_logging(
    level: str = "INFO",
    *,
    log_dir: str | Path | None = None,
    filename: str = "kira.log",
    console: bool = True,
    json_console: bool = False,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    session: str | None = None,
) -> logging.Logger:
    """Configure the root logger. Idempotent: re-calling retunes, it never duplicates."""
    root = logging.getLogger()
    root.setLevel(_level(level))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    if console:
        handler = logging.StreamHandler(stream=sys.stdout if sys.stdout else sys.stderr)
        if json_console:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_CONSOLE_DATEFMT))
        root.addHandler(handler)

    if log_dir is not None:
        directory = Path(log_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            root.warning("cannot create log dir %s: %s", directory, exc)
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                directory / filename,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True,
            )
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)

    for noisy in _NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if session:
        root.info("session start", extra={"session": session})
    return root


def _level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def get_handlers() -> list[logging.Handler]:
    return list(logging.getLogger().handlers)


def get_log_dir() -> Path | None:
    """The directory the file handler writes to, if any."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            return Path(handler.baseFilename).parent
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def capture(level: int = logging.DEBUG) -> contextlib.AbstractContextManager[list[logging.LogRecord]]:
    """Collect log records as a list, for tests and one-shot diagnostics."""
    records: list[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Sink(level)
    root = logging.getLogger()
    previous = root.level
    root.addHandler(handler)
    root.setLevel(min(previous, level))
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)


def new_session_id() -> str:
    """A short id shared by every line of one process, stable for the lifetime."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{os.getpid() & 0xFFFF:04x}"


def format_latency(ms: float) -> str:
    """Render milliseconds for HUD status lines."""
    if ms < 1:
        return f"{ms * 1000:.0f}us"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60_000:.1f}m"


def truncate(text: str, limit: int = 400) -> str:
    """Trim for logs while preserving how much was cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit}]"


def truncate_bytes(data: bytes, limit: int = 64) -> str:
    """A log-safe short representation of binary data."""
    shown = data[:limit]
    hexed = shown.hex()
    if len(data) > limit:
        hexed += "…"
    return f"{hexed} ({len(data)} bytes)"


def exc_one_line(exc: BaseException) -> str:
    """A single-line exception summary for status displays."""
    return f"{type(exc).__name__}: {exc}"
