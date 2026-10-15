"""Verification tools for exercising the tool-calling path end to end.

These exist to let ``tests/test_llm_tools.py`` prove that a local model can actually
drive the agent loop — schema, dispatch, error recovery, and the spoken follow-up —
without the ArtCraft suite being built yet. They are intentionally small and depend on
nothing beyond the standard library.

``get_system_stats`` reads real machine state. CPU and memory come from Windows kernel
counters through ``ctypes``, so no profiling package is required; GPU utilisation is
best-effort via ``nvidia-smi`` and reports itself as unavailable rather than guessing.

``artcraft_canvas_action`` keeps a canvas in process memory. It is a stand-in for the
real ArtCraft tool, not a preview of it — the action set and error semantics are what
the LLM integration contract depends on.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time

from core.tools.registry import Registry, Tool, ToolError


# --------------------------------------------------------------------------- #
# Machine telemetry
# --------------------------------------------------------------------------- #

#: Previous (timestamp, idle_ticks, total_ticks). Kept as a delta source so repeated
#: calls report utilisation over a real window instead of one instantaneous reading.
_CPU_BASELINE: dict[str, object] = {}
_CPU_LOCK = threading.Lock()


def _windows_cpu_percent(settle_s: float = 0.05) -> float | None:
    """CPU utilisation from kernel time counters. No third-party dependency."""
    if os.name != "nt":
        return None
    try:
        import ctypes
    except ImportError:  # pragma: no cover
        return None

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    idle, kernel, user = (ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64())
    if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    # kernel already includes user time, so kernel is the total busy+idle window.
    total = int(kernel.value)
    now = time.monotonic()
    idle_v, total_v = int(idle.value), total

    with _CPU_LOCK:
        prev_idle = _CPU_BASELINE.get("idle")
        prev_total = _CPU_BASELINE.get("total")
        prev_at = _CPU_BASELINE.get("at")

        if prev_idle is not None and prev_total is not None and prev_at is not None:
            elapsed = now - float(prev_at)
            dt_total = total_v - int(prev_total)
            dt_idle = idle_v - int(prev_idle)
            if elapsed >= settle_s and dt_total > 0:
                # A real window: the reading reflects actual recent load.
                return round(max(0.0, min(100.0, (dt_total - dt_idle) / dt_total * 100.0)), 1)

        _CPU_BASELINE.update({"idle": idle_v, "total": total_v, "at": now})
        # First sample, or sampled too soon to be meaningful: report the cumulative
        # figure since boot. Always available and honest, just coarse.
        return round((total_v - idle_v) / total_v * 100.0, 1) if total_v else 0.0


def _proc_stat_cpu_percent(settle_s: float = 0.05) -> float | None:
    """Linux fallback: parse /proc/stat, same delta approach as the Windows path."""
    try:
        with open("/proc/stat", encoding="ascii") as fh:
            parts = fh.readline().split()
    except OSError:  # pragma: no cover
        return None
    if len(parts) < 5:
        return None
    try:
        values = [int(p) for p in parts[1:]]
    except ValueError:  # pragma: no cover
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    now = time.monotonic()

    prev = _CPU_BASELINE.get("linux")
    if prev is not None and now - prev[2] >= settle_s:
        dt_total = total - prev[0]
        dt_idle = idle - prev[1]
        if dt_total > 0:
            return round(max(0.0, min(100.0, (dt_total - dt_idle) / dt_total * 100.0)), 1)
    _CPU_BASELINE["linux"] = (idle, total, now)
    return round((total - idle) / total * 100.0, 1) if total else 0.0


def _memory_info() -> dict:
    """Physical memory from GlobalMemoryStatusEx on Windows, /proc/meminfo elsewhere."""
    if os.name == "nt":
        try:
            import ctypes
        except ImportError:  # pragma: no cover
            return {}

        class _STATMEM(ctypes.Structure):
            _fields_ = (
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            )

        stat = _STATMEM()
        stat.dwLength = ctypes.sizeof(_STATMEM)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return {}
        total = int(stat.ullTotalPhys)
        used = total - int(stat.ullAvailPhys)
        return {
            "total_gb": round(total / 1024**3, 2),
            "used_gb": round(used / 1024**3, 2),
            "percent_used": round(int(stat.dwMemoryLoad), 1),
        }

    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            raw = {ln.split(":")[0]: int(ln.split()[1]) for ln in fh if ":" in ln}
    except OSError:  # pragma: no cover
        return {}
    total_kb = raw.get("MemTotal", 0)
    available_kb = raw.get("MemAvailable", raw.get("MemFree", 0))
    used_kb = max(0, total_kb - available_kb)
    if not total_kb:
        return {}
    return {
        "total_gb": round(total_kb / 1024**2, 2),
        "used_gb": round(used_kb / 1024**2, 2),
        "percent_used": round(used_kb / total_kb * 100.0, 1),
    }


def _gpu_info() -> dict:
    """GPU utilisation via nvidia-smi when present. Never raises, never guesses."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"available": False, "note": "nvidia-smi not found; no NVIDIA GPU telemetry"}
    try:
        out = subprocess.run(
            [
                binary, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=4, check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "note": f"nvidia-smi failed: {type(exc).__name__}"}

    gpus: list[dict] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_percent": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_total_mib": int(parts[4]),
            })
        except ValueError:  # pragma: no cover
            continue
    return {"available": bool(gpus), "gpus": gpus}


def _collect_stats() -> dict:
    # Sampling window for the CPU delta. Without this the first call reports
    # utilisation over a window of microseconds, which reads as essentially 0%.
    cpu = _windows_cpu_percent(settle_s=0.05)
    if cpu is None:
        cpu = _proc_stat_cpu_percent(settle_s=0.05)
    return {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cores": os.cpu_count(),
        },
        "cpu": {
            "utilization_percent": cpu,
            "note": "delta since last sample; cumulative since boot on first call"
                    if cpu is not None else "unavailable",
        },
        "memory": _memory_info() or {"available": False},
        "gpu": _gpu_info(),
        "timestamp_epoch": round(time.time(), 1),
    }


# --------------------------------------------------------------------------- #
# ArtCraft canvas (in-memory stand-in)
# --------------------------------------------------------------------------- #

CANVAS_ACTIONS = ("new_canvas", "select_tool", "export")
CANVAS_TOOLS = ("brush", "eraser", "line", "shape", "text", "select")


class CanvasState:
    """A minimal in-memory canvas so the mock behaves like a real stateful tool."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.width = 1024
        self.height = 768
        self.tool = "brush"
        self.size = 4
        self.strokes: list[dict] = []
        self.name = "untitled"
        self.modified = False

    def summary(self) -> dict:
        return {
            "name": self.name,
            "size": f"{self.width}x{self.height}",
            "tool": self.tool,
            "stroke_count": len(self.strokes),
            "modified": self.modified,
        }


#: Single shared instance: the model expects canvas state to persist across calls.
CANVAS = CanvasState()


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

_SYSTEM_STATS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "include_gpu": {
            "type": "boolean",
            "description": "Also query GPU utilisation via nvidia-smi.",
            "default": True,
        },
    },
    "required": [],
}

_CANVAS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": "The canvas operation to perform.",
            "enum": list(CANVAS_ACTIONS),
        },
        "parameters": {
            "type": "object",
            "description": (
                "Action-specific settings: new_canvas takes width/height/name; "
                "select_tool takes tool/size; export takes format/path/force."
            ),
            "additionalProperties": True,
        },
    },
    "required": ["action"],
}


def _canvas_action(action: str, parameters: dict | None = None) -> dict:
    """One ArtCraft canvas operation. Raises ToolError on any invalid input."""
    if action not in CANVAS_ACTIONS:
        raise ToolError(f"unknown action {action!r}. Valid actions: {', '.join(CANVAS_ACTIONS)}")
    params = parameters or {}
    if not isinstance(params, dict):
        raise ToolError(f"parameters must be an object, got {type(params).__name__}")

    with CANVAS._lock:
        if action == "new_canvas":
            width = int(params.get("width", CANVAS.width))
            height = int(params.get("height", CANVAS.height))
            name = str(params.get("name", "untitled"))
            if not (1 <= width <= 8192 and 1 <= height <= 8192):
                raise ToolError(f"canvas dimensions {width}x{height} out of range 1-8192")
            CANVAS.reset()
            CANVAS.width, CANVAS.height, CANVAS.name = width, height, name
            return {"action": action, "status": "created", "canvas": CANVAS.summary()}

        if action == "select_tool":
            tool = str(params.get("tool", "")).lower()
            if tool not in CANVAS_TOOLS:
                raise ToolError(f"unknown tool {tool!r}. Valid: {', '.join(CANVAS_TOOLS)}")
            size = params.get("size")
            if size is not None:
                size = int(size)
                if not 1 <= size <= 500:
                    raise ToolError(f"brush size {size} out of range 1-500")
                CANVAS.size = size
            CANVAS.tool = tool
            return {"action": action, "status": "selected", "canvas": CANVAS.summary()}

        # action == "export"
        fmt = str(params.get("format", "png")).lower()
        if fmt not in ("png", "jpg", "jpeg"):
            raise ToolError(f"unsupported export format {fmt!r}; use png, jpg, or jpeg")
        target = str(params.get("path", "")) or f"{CANVAS.name}.{fmt}"
        if CANVAS.modified is False and not params.get("force"):
            raise ToolError("canvas is not modified; nothing to export (pass force=true)")
        # The real ArtCraft suite writes bytes; this stand-in reports what it would do.
        return {
            "action": action,
            "status": "queued",
            "target": target,
            "format": fmt,
            "canvas": CANVAS.summary(),
        }


def _system_stats_tool(include_gpu: bool = True) -> dict:
    """Current CPU load, physical memory usage, and GPU utilisation if available.

    Use this when the user asks how the machine is doing, how much RAM is free,
    or whether the GPU is busy.
    """
    stats = _collect_stats()
    if not include_gpu:
        # Report exactly what would be available, without the extra query cost.
        stats["gpu"] = {"available": False, "note": "skipped on request"}
    return stats


def _canvas_tool(action: str, parameters: dict | None = None) -> dict:
    """Perform one ArtCraft canvas operation.

    Actions: new_canvas (start fresh), select_tool (pick brush/eraser/etc.),
    export (save the canvas).
    """
    return _canvas_action(action, parameters)


def register_verification_tools(registry: Registry) -> None:
    """Add the two verification tools to ``registry``."""
    registry.register(Tool(
        name="get_system_stats",
        description=(
            "Read current machine telemetry: CPU utilisation, physical memory in use, "
            "and GPU load when an NVIDIA driver is present. Read-only."
        ),
        fn=_system_stats_tool,
        parameters=_SYSTEM_STATS_SCHEMA,
        timeout_s=10.0,
        tags=("system", "read"),
    ))
    registry.register(Tool(
        name="artcraft_canvas_action",
        description=(
            "Drive the ArtCraft drawing canvas: start a new canvas, select a drawing "
            "tool, or export the current canvas."
        ),
        fn=_canvas_tool,
        parameters=_CANVAS_SCHEMA,
        timeout_s=10.0,
        tags=("artcraft", "write"),
    ))


__all__ = [
    "CANVAS",
    "CANVAS_ACTIONS",
    "CANVAS_TOOLS",
    "register_verification_tools",
]
