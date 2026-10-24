"""HUD widgets for the tri-column tactical dashboard.

These are deliberately thin views. They hold no pipeline state and take no settings:
the window owns the conversation and pushes text down. That keeps them easy to test
and free of the audio/LLM lifecycle.

Layout ownership: ``MainWindow`` owns the columns; each panel here owns only its own
header, meters and labels.

All metrics come from a sampler callable that returns a dict, so the panels never
import a system-monitoring library. ``sample_metrics()`` at the bottom of this file is
that sampler for a bare process: it uses only the standard library and ctypes, so the
HUD works with or without ``psutil`` installed.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import time

from PyQt6.QtCore import QDateTime, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

from ui.theme import (
    Palette,
    RoundedCard,
    StatusDot,
    UI_MONOSPACE,
    _with_alpha,
    mono_font,
)


# --------------------------------------------------------------------------- #
# Panel frame
# --------------------------------------------------------------------------- #

class Panel(RoundedCard):
    """A titled tech-glass panel. Angular: radius 0 and corner brackets."""

    def __init__(
        self,
        palette: Palette,
        *,
        title: str = "",
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(palette, radius=0)
        self._palette = palette
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QWidget()
        hlayout = QHBoxLayout(header)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(8)

        self._dot = StatusDot(palette, size=8)
        self._dot.set_state(palette.accent, pulse=False)
        hlayout.addWidget(self._dot)

        self._title = QLabel(title)
        self._title.setProperty("tech", True)
        hlayout.addWidget(self._title)

        self._subtitle = QLabel(subtitle)
        self._subtitle.setProperty("accent2", True)
        hlayout.addWidget(self._subtitle)
        hlayout.addStretch(1)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {_with_alpha(palette.accent, 60).name()};")
        hlayout.addWidget(divider)

        layout.addWidget(header)
        self.content = QVBoxLayout()
        self.content.setSpacing(8)
        layout.addLayout(self.content, 1)
        self.layout = layout

    def set_status(self, text: str, *, color: str | None = None, pulse: bool = False) -> None:
        """Set the header dot colour and optional subtitle."""
        self._dot.set_state(color or self._palette.accent, pulse=pulse)
        if text:
            self._subtitle.setText(text)


# --------------------------------------------------------------------------- #
# Waveform
# --------------------------------------------------------------------------- #

class Waveform(QWidget):
    """A live level meter.

    Peaks arrive as floats in roughly ``[-1, 1]``. The widget keeps a ring buffer and
    draws mirrored bars, relaxing each frame toward zero so silence visibly returns to
    zero even if the caller stops sending peaks. ``set_target`` animates a synthetic
    pulse, useful while waiting for speech to end.
    """

    BARS = 28

    def __init__(self, palette: Palette, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._peaks = [0.0] * self.BARS
        self._decay = 0.90
        self._target: float | None = None
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps is plenty for a meter
        self._timer.timeout.connect(self._tick)

    def push(self, level: float) -> None:
        """Record a new peak. Clamped, since a hot microphone can exceed unity."""
        clamped = min(1.0, max(0.0, abs(float(level))))
        if clamped > self._peaks[-1]:
            self._peaks = self._peaks[1:] + [clamped]
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def set_target(self, value: float | None) -> None:
        """Drive the meter synthetically, e.g. a pulse while listening."""
        self._target = value
        if value is not None and not self._timer.isActive():
            self._timer.start()

    def clear(self) -> None:
        self._peaks = [0.0] * self.BARS
        self._target = None
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        """Relax the meter toward zero; stop the timer once it is fully still."""
        alive = False
        if self._target is not None:
            self._target = max(0.0, self._target - 0.02)
            if self._target <= 0.01:
                self._target = None
            else:
                alive = True
        for i, peak in enumerate(self._peaks):
            if peak > 0.002:
                value = peak * self._decay
                if value < 0.002:
                    value = 0.0
                if value != peak:
                    self._peaks[i] = value
                    alive = True
        self.update()
        if not alive:
            self._timer.stop()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        mid = height / 2.0
        gap = 3.0
        bar_w = max(1.0, (width - gap * (self.BARS - 1)) / self.BARS)
        centre = height * 0.05  # minimum bar height, so silence is still visible

        peaks = list(self._peaks)
        if self._target is not None:
            envelope = (math.sin(time.time() * 4.0) + 1.0) * 0.5 * 0.65 + 0.35
            peaks = [max(p, self._target * envelope) for p in peaks]

        for i, peak in enumerate(peaks):
            bar_h = max(centre, peak * (height - centre * 1.5))
            x = i * (bar_w + gap)
            colour = QColor(self._palette.accent)
            colour.setAlpha(int(120 + 135 * peak))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            radius = min(bar_w / 2.0, bar_h / 2.0)
            painter.drawRoundedRect(x, mid - bar_h / 2.0, bar_w, bar_h, radius, radius)

        faint = QColor(self._palette.text_faint)
        faint.setAlpha(70)
        painter.setPen(QPen(faint, 1))
        painter.drawLine(0, int(mid), width, int(mid))
        painter.end()


# --------------------------------------------------------------------------- #
# System panel
# --------------------------------------------------------------------------- #

class Meter(QProgressBar):
    """A continuous fill bar for CPU / MEM / AI gauges.

    Coloured by value: below 60 % is warning amber, at and above it turns critical
    red. Uses Fusion style to bypass Windows native chunking.
    """

    def __init__(self, palette: Palette, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        # Force Fusion style to bypass Windows native progress bar chunking
        self.setStyle(QStyleFactory.create('Fusion'))
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(0)
        self.setTextVisible(False)
        self.setFixedHeight(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, percent: float) -> None:
        value = int(min(100.0, max(0.0, float(percent))))
        self.setValue(value)
        # Update color based on value threshold
        if value < 60:
            color = self._palette.warning
        else:
            color = self._palette.danger
        # Apply inline stylesheet for dynamic coloring
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid rgba(0, 229, 255, 0.2);
                background-color: #071426;
                border-radius: 2px;
                text-align: right;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
                margin: 0px;
                width: 1px;
            }}
        """)


def _stat_row(palette: Palette, label: str) -> tuple[Meter, QLabel, QWidget]:
    meter = Meter(palette)
    value = QLabel("-- %")
    value.setProperty("tech", True)
    value.setFixedWidth(56)
    value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    lab = QLabel(label)
    lab.setProperty("tech", True)
    lab.setFixedWidth(38)
    layout.addWidget(lab)
    layout.addWidget(meter, 1)
    layout.addWidget(value)
    return meter, value, row


class SystemPanel(Panel):
    """CPU / MEM / AI gauges plus a pulsing digital clock.

    Metrics come from a callback the window supplies, sampled on a ``QTimer``. A failed
    sampler just leaves the previous value on screen -- a missing gauge is worse UX than
    a stale one, and a raise here would take the whole window down.
    """

    def __init__(
        self,
        palette: Palette,
        *,
        sampler=None,
        interval: int = 2000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(palette, title="SYSTEM // STATUS", subtitle="[ DIAGNOSTICS ]")
        self._palette = palette
        self._sampler = sampler if sampler is not None else sample_metrics()

        self.cpu_meter, self.cpu_label, self.cpu_row = _stat_row(palette, "CPU")
        self.mem_meter, self.mem_label, self.mem_row = _stat_row(palette, "MEM")
        self.ai_meter, self.ai_label, self.ai_row = _stat_row(palette, "AI")
        self.content.addWidget(self.cpu_row)
        self.content.addWidget(self.mem_row)
        self.content.addWidget(self.ai_row)

        self.clock = QLabel("--:--:--")
        self.clock.setProperty("telemetry", True)
        self.clock.setProperty("bright", True)
        self.clock.setMinimumHeight(20)
        self.content.addWidget(self.clock)

        self._timer = QTimer(self)
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    @property
    def timer(self) -> QTimer:
        return self._timer

    def set_sampler(self, sampler) -> None:
        """Override the metric source. Called once the window wires the LLM in."""
        if sampler is not None:
            self._sampler = sampler

    def sample(self) -> dict:
        """One CPU / MEM / AI reading. Never raises; returns {} when unknown."""
        if self._sampler is None:
            return {}
        try:
            reading = self._sampler()
        except Exception:
            return {}
        return reading if isinstance(reading, dict) else {}

    def _tick(self) -> None:
        reading = self.sample()

        now = QDateTime.currentDateTime()
        self.clock.setText(now.toString("HH:mm:ss") + " // " + now.toString("dd MMM yyyy"))
        wave = (math.sin(now.time().msec() / 1000.0 * math.tau) + 1.0) * 0.5
        self.clock.setStyleSheet(
            f"color: {self._palette.accent}; font-size: 10pt; "
            f"letter-spacing: 0.8px; opacity: {0.72 + 0.28 * wave:.2f};"
        )

        if not reading:
            return
        cpu = float(reading.get("cpu", 0.0) or 0.0)
        mem_pct = float(reading.get("mem_pct", 0.0) or 0.0)
        ai = float(reading.get("ai", reading.get("gpu", 0.0)) or 0.0)
        self.cpu_meter.set_value(cpu)
        self.mem_meter.set_value(mem_pct)
        self.ai_meter.set_value(ai)
        self.cpu_label.setText(f"{cpu:5.0f} %")
        self.mem_label.setText(f"{mem_pct:5.0f} %")
        self.ai_label.setText(f"{ai:5.0f} %")


# --------------------------------------------------------------------------- #
# System sampling
# --------------------------------------------------------------------------- #

class _CpuSampler:
    """Process CPU % from ``time.process_time`` deltas.

    ``os.times().elapsed`` is the tempting one-liner, but on Windows it resolves
    through a real-time-clock fallback and reports the same value while busy as
    while idle. ``time.process_time`` reads the actual thread CPU clock and is
    the correct counter. First call warms the baseline and reports 0.0.
    """

    def __init__(self) -> None:
        self._last_ns: int | None = None
        self._last_wall = 0.0
        self._cores = max(1, os.cpu_count() or 1)

    def __call__(self) -> float:
        try:
            ns = time.process_time_ns()
            wall = time.perf_counter()
        except (OSError, ValueError):
            return 0.0
        if self._last_ns is None:
            self._last_ns, self._last_wall = ns, wall
            return 0.0
        delta_wall = wall - self._last_wall
        if delta_wall <= 0:
            return 0.0
        used_s = (ns - self._last_ns) / 1e9
        self._last_ns, self._last_wall = ns, wall
        # Divided by the core count, so 100 % is one fully busy core.
        return min(100.0, used_s / delta_wall / self._cores * 100.0)


def _rss_mb() -> float:
    """This process's working set in MB, zero when the OS refuses to say."""
    try:
        if platform.system() == "Windows":
            class _PMC(ctypes.Structure):
                _fields_ = (
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                )

            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            stat = _PMC()
            stat.cb = ctypes.sizeof(stat)
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            rc = psapi.GetProcessMemoryInfo(
                ctypes.c_void_p(ctypes.WinDLL("kernel32", use_last_error=True).GetCurrentProcess()),
                ctypes.byref(stat),
                ctypes.c_ulong(stat.cb),
            )
            return stat.WorkingSetSize / (1024 * 1024) if rc else 0.0
        with open(f"/proc/{os.getpid()}/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * _os_pagesize() / (1024 * 1024)
    except Exception:
        return 0.0


def _os_pagesize() -> int:
    try:
        return os.sysconf("SC_PAGESIZE")
    except (ValueError, OSError):
        return 4096


def _total_memory_mb() -> float:
    """Total physical RAM. ctypes on Windows, /proc/meminfo elsewhere."""
    try:
        if platform.system() == "Windows":
            class _MemoryStat(ctypes.Structure):
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

            stat = _MemoryStat()
            stat.dwLength = ctypes.sizeof(_MemoryStat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 * 1024)
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


class _GpuSampler:
    """GPU utilisation via DXGI, or 0.0 wherever there is no NVIDIA adapter.

    AMD iGPUs have no DXGI counter exposed through pynvml, so this reads as
    zero on hardware like the 5600U rather than inventing a number. It is
    present only when pynvml imports; anything else falls through to 0.0.
    """

    def __init__(self) -> None:
        self._utilisation = 0
        self._handle = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            if handle is not None:
                self._handle = handle
        except Exception:
            self._handle = None

    def __call__(self) -> float:
        if self._handle is None:
            return 0.0
        try:
            import pynvml  # type: ignore

            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu)
        except Exception:
            return 0.0


def sample_metrics():
    """Build a sampler returning ``cpu``, ``mem_pct``, ``mem_mb``, ``gpu``.

    The dict is consumed by :class:`SystemPanel`; anything missing is simply
    absent and the corresponding meter is left untouched.
    """
    cpu = _CpuSampler()
    gpu = _GpuSampler()
    total = _total_memory_mb()
    # Sample a few readings so the first real value is not a cold baseline.
    cpu()

    def _read() -> dict:
        reading = {
            "cpu": cpu(),
            "gpu": gpu(),
        }
        used_mb = _rss_mb()
        reading["mem_mb"] = round(used_mb, 1)
        reading["mem_pct"] = round(used_mb / total * 100.0, 1) if total > 0 else 0.0
        return reading

    return _read


# --------------------------------------------------------------------------- #
# KIRA core panel
# --------------------------------------------------------------------------- #

def _kv(palette: Palette, key: str, value: str = "", *, bright: bool = False) -> tuple[QWidget, QLabel]:
    """One ``KEY: value`` line. Returns the row widget plus the value label."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    k = QLabel(key)
    k.setProperty("tech", True)
    v = QLabel(value)
    v.setProperty("telemetry" if bright else "tech", True)
    layout.addWidget(k)
    layout.addWidget(v, 1)
    return row, v


class KiraCorePanel(Panel):
    """Connectivity read-out for the local LLM server."""

    @staticmethod
    def _host_only(url: str) -> str:
        """``http://127.0.0.1:11434/v1`` -> ``127.0.0.1:11434``.

        The spec shows the link as a host:port pair; the scheme and any trailing
        path are noise on the HUD.
        """
        text = (url or "").strip()
        if not text:
            return "127.0.0.1:11434"
        text = text.split("://", 1)[-1]
        return text.split("/", 1)[0] or "127.0.0.1:11434"

    def __init__(self, palette: Palette, *, host: str = "", model: str = "", parent=None) -> None:
        super().__init__(palette, title="KIRA // CORE", subtitle="[ NEURAL LINK ]")
        self._host = host
        self._model = model
        host_view = self._host_only(host)

        rows = [
            _kv(palette, "CONNECTED:", "ONLINE", bright=True),
            _kv(palette, "LOCAL HOST:", host_view, bright=True),
            _kv(palette, "CONNECTED MODEL:", model, bright=True),
            _kv(palette, "THROUGHPUT:", "-- t/s"),
            _kv(palette, "LATENCY:", "-- s"),
        ]
        for row, value in rows:
            self.content.addWidget(row)
        self.online = rows[0][1]
        self.host_label = rows[1][1]
        self.model_label = rows[2][1]
        self.tps = rows[3][1]
        self.latency = rows[4][1]

    def set_host(self, text: str) -> None:
        self._host = text
        self.host_label.setText(self._host_only(text))

    def set_model(self, text: str) -> None:
        self._model = text
        self.model_label.setText(text)

    def set_link(self, *, online: bool, note: str = "") -> None:
        self.online.setText("ONLINE" if online else note or "OFFLINE")
        self.set_status(
            "[ ACTIVE ]" if online else f"[ {note.upper() or 'OFFLINE'} ]",
            color=self._palette.success if online else self._palette.warning,
        )

    def set_metrics(self, *, tps: float = 0.0, latency_s: float = 0.0) -> None:
        self.tps.setText(f"{tps:.1f} t/s" if tps > 0 else "-- t/s")
        self.latency.setText(f"{latency_s:.1f} s" if latency_s > 0 else "-- s")

    def reset_metrics(self) -> None:
        self.set_metrics()


# --------------------------------------------------------------------------- #
# Project KIRA panel
# --------------------------------------------------------------------------- #

class ProjectKiraPanel(Panel):
    """Repository and telemetry spec readout."""

    def __init__(self, palette: Palette, *, host: str = "", model: str = "", parent=None) -> None:
        super().__init__(palette, title="PROJECT KIRA", subtitle="[ TELEMETRY SPECS ]")

        header = QLabel("TELEMETRY SPECS:")
        header.setProperty("tech", True)
        header.setStyleSheet(f"color: {palette.accent_2};")
        self.content.addWidget(header)

        rows = [
            _kv(palette, "REPOSITORY CHECK:", "artcraft-suite / neural_mesh"),
            _kv(palette, "HOST:", host),
            _kv(palette, "MODEL:", model),
            _kv(palette, "TTS ENGINE:", "--"),
            _kv(palette, "AUDIO:", "--"),
            _kv(palette, "ACTIVE TOOLS:", "--", bright=True),
        ]
        for row, value in rows:
            self.content.addWidget(row)
        self.specs = header
        self.repo = rows[0][1]
        self.host_label = rows[1][1]
        self.model_label = rows[2][1]
        self.engine = rows[3][1]
        self.device = rows[4][1]
        self.tools = rows[5][1]

    def set_host(self, text: str) -> None:
        self.host_label.setText(text)

    def set_model(self, text: str) -> None:
        self.model_label.setText(text)

    def set_engine(self, text: str) -> None:
        self.engine.setText(text)

    def set_device(self, text: str) -> None:
        self.device.setText(text)

    def set_tools(self, count: int, names: list[str] | None = None) -> None:
        if names:
            shown = ", ".join(names[:3])
            if len(names) > 3:
                shown += f" …[+{len(names) - 3}]"
            self.tools.setText(f"[{count}] // {shown}")
        else:
            self.tools.setText(f"[{count}]")


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #

def _row(
    text: str,
    *,
    palette: Palette,
    who: str,
    muted: bool,
    assistant: bool = False,
    label: str = "",
) -> RoundedCard:
    card = RoundedCard(palette, radius=4)
    if assistant and not muted:
        # Lean the assistant surface slightly toward the accent; others stay glass.
        card.set_tint(palette.glass, 210)
    layout = QHBoxLayout(card)
    layout.setContentsMargins(12, 8, 12, 8)
    layout.setSpacing(10)

    dot = StatusDot(palette, size=8)
    dot.set_state(
        palette.text_faint if muted else (palette.accent if assistant else palette.accent_bright),
        pulse=False,
    )

    body = QVBoxLayout()
    body.setSpacing(3)
    caption = QLabel(label or (who.upper() if who else "AGENT"))
    # Assistant rows read as a lit protocol header; everyone else as quiet telemetry.
    if assistant and not muted:
        caption.setProperty("caption", True)
    else:
        caption.setProperty("tech", True)
    message = QLabel(text)
    message.setWordWrap(True)
    message.setProperty("assistant" if assistant and not muted else "user", True)
    if muted:
        message.setProperty("dim", True)
    body.addWidget(caption)
    body.addWidget(message)

    layout.addWidget(dot)
    layout.addLayout(body, 1)
    card.message_label = message  # streamed into by Transcript.append_text
    return card


class Transcript(QScrollArea):
    """A scrolling record of the exchange. Old rows are dropped to bound memory."""

    line_added = pyqtSignal()

    def __init__(self, palette: Palette, *, parent: QWidget | None = None, max_rows: int = 200) -> None:
        super().__init__(parent)
        self._palette = palette
        self._max_rows = max_rows
        self._rows: list[RoundedCard] = []

        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(2, 2, 6, 2)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)  # keeps new content pinned above this
        self.setWidget(container)

        self._streaming: RoundedCard | None = None

    def _prune(self) -> None:
        while len(self._rows) > self._max_rows:
            old = self._rows.pop(0)
            self._layout.removeWidget(old)
            old.deleteLater()
            if self._streaming is old:
                # Deleted the bubble we were writing into: drop the reference so a
                # later chunk starts a fresh row instead of touching dead memory.
                self._streaming = None

    def add(
        self,
        text: str,
        *,
        who: str = "you",
        muted: bool = False,
        label: str = "",
        assistant: bool = False,
    ) -> RoundedCard:
        """Append one exchange. Returns the widget so the caller can animate it.

        ``label`` overrides the caption shown above the message; ``assistant`` marks
        the row as an AI-originated one for dot colour and caption styling.
        """
        card = _row(text, palette=self._palette, who=who, muted=muted, assistant=assistant, label=label)
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._rows.append(card)
        self._prune()
        self._scroll_to_bottom()
        self.line_added.emit()
        return card

    def append_text(self, text: str, *, who: str = "assistant", label: str = "") -> RoundedCard:
        """Stream text into the last assistant row instead of making a new bubble.

        The first call creates the row; later calls extend it in place, so the HUD
        shows tokens as they arrive rather than jumping to the complete sentence.
        """
        card = self._streaming
        if card is None:
            card = _row("", palette=self._palette, who=who, muted=False, assistant=True, label=label)
            self._layout.insertWidget(self._layout.count() - 1, card)
            self._rows.append(card)
            self._prune()
            self._streaming = card
            self.line_added.emit()
        card.message_label.setText(text)
        self._scroll_to_bottom()
        return card

    def clear_streaming(self) -> None:
        """Close the open bubble once the turn is complete."""
        self._streaming = None

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear(self) -> None:
        self._streaming = None
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    @property
    def row_count(self) -> int:
        return len(self._rows)


# --------------------------------------------------------------------------- #
# Terminal log
# --------------------------------------------------------------------------- #

class TerminalLog(QWidget):
    """A bounded, monospaced command log.

    Lines are plain text appended at the bottom, oldest dropped first. The prompt
    cursor is a separate label so it can blink on a timer without a layout pass.
    """

    PROMPT = "KIRA $"

    def __init__(self, palette: Palette, *, parent: QWidget | None = None, max_lines: int = 400) -> None:
        super().__init__(parent)
        self._palette = palette
        self._max_lines = max_lines
        self._lines: list[str] = []
        self._on = True

        self._font = mono_font(9)
        self._view = QLabel(self)
        self._view.setFont(self._font)
        self._view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._view.setStyleSheet(
            f"color: {palette.text_dim}; font-family: {UI_MONOSPACE}; "
            f"font-size: 9pt; letter-spacing: 0.3px;"
        )

        self._cursor = QLabel(" ", self)
        self._cursor.setFont(self._font)
        self._cursor.setStyleSheet(f"color: {palette.accent}; background: transparent;")
        self._cursor.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._cursor.setFixedHeight(14)
        self._cursor.setFixedWidth(8)

        self._timer = QTimer(self)
        self._timer.setInterval(530)
        self._timer.timeout.connect(self._blink)
        self._timer.start()

    @property
    def timer(self) -> QTimer:
        return self._timer

    def add(self, text: str) -> None:
        """Append a line. Newlines are split so one call can log a whole message."""
        lines = [part for part in str(text).splitlines() if part.strip()] or [" "]
        self._lines.extend(lines)
        while len(self._lines) > self._max_lines:
            self._lines.pop(0)
        self._view.setText("\n".join(self._lines))
        self._relayout()

    def clear(self) -> None:
        self._lines = []
        self._view.setText("")

    def _relayout(self) -> None:
        view = self._view
        view.resize(max(1, self.width() - 22), max(1, self.height() - 26))
        view.move(10, 6)
        self._cursor.move(4, max(1, self.height() - 18))

    def _blink(self) -> None:
        self._on = not self._on
        self._cursor.setVisible(self._on)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Dark base: monospace logs need contrast against the translucent panel.
        bg = QColor(self._palette.window_bg)
        bg.setAlpha(225)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(0, 0, self.width(), self.height())

        # A faint scanline wash, one alpha step, kept deliberately sparse.
        line = QColor(self._palette.accent)
        line.setAlpha(8)
        painter.setPen(QPen(line, 1))
        y = 4
        while y < self.height():
            painter.drawLine(0, y, self.width(), y)
            y += 4

        # Corner brackets, matching the panels so the log reads as part of the HUD.
        br = QColor(self._palette.border)
        br.setAlpha(120)
        painter.setPen(QPen(br, 1.4))
        length = 11.0
        for x, y, sx, sy in (
            (0.0, 0.0, 1, 1),
            (float(self.width()), 0.0, -1, 1),
            (0.0, float(self.height()), 1, -1),
            (float(self.width()), float(self.height()), -1, -1),
        ):
            painter.drawLine(int(x), int(y), int(x + sx * length), int(y))
            painter.drawLine(int(x), int(y), int(x), int(y + sy * length))
        painter.end()


# --------------------------------------------------------------------------- #
# Status strip
# --------------------------------------------------------------------------- #

def _div(palette: Palette) -> QWidget:
    """A literal `` // `` separator, so the strip reads as one line, not a row."""
    label = QLabel(" // ")
    label.setProperty("accent2", True)
    label.setFixedWidth(26)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class StatusStrip(QWidget):
    """Tactical telemetry readout: core, throughput, latency, context.

    Fields are monospace and joined with literal `` // `` dividers so the strip reads
    as a single status line. Every field carries a real default -- an empty label
    would make the strip look broken until the first turn.
    """

    DIV = " // "

    def __init__(self, palette: Palette, *, model: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._model = model
        self._dot = StatusDot(palette, size=9)
        self._dot.set_state(palette.success, pulse=False)

        def _field(text: str, *, bright: bool = False) -> QLabel:
            label = QLabel(text)
            label.setProperty("telemetry" if bright else "tech", True)
            if bright:
                label.setProperty("bright", True)
            return label

        self.state_label = _field(f"STATE: {self._initial_state}", bright=True)
        self.model_label = _field(
            f"CORE: [{self._model}]" if self._model else "CORE: [—]", bright=True
        )
        self.device_label = _field("DEV: —")
        self.tps_label = _field("SPEED: ⚡ -- t/s")
        self.latency_label = _field("LATENCY: ⏱️ --s")
        self.token_label = _field("CONTEXT: 🧠 -- tok")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._dot)
        layout.addWidget(self.state_label)
        layout.addWidget(_div(palette))
        layout.addWidget(self.model_label)
        layout.addSpacerItem(QSpacerItem(6, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        layout.addWidget(self.device_label)
        layout.addWidget(_div(palette))
        layout.addWidget(self.tps_label)
        layout.addWidget(_div(palette))
        layout.addWidget(self.latency_label)
        layout.addWidget(_div(palette))
        layout.addWidget(self.token_label)
        self.setVisible(True)

    @property
    def _initial_state(self) -> str:
        return "ready"

    @property
    def state(self) -> str:
        """The raw state word, unprefixed -- callers compare against this."""
        text = self.state_label.text()
        return text.split(":", 1)[1].strip() if ":" in text else text

    def set_state(self, text: str, *, color: str | None = None, pulse: bool = False) -> None:
        """Update the activity readout and its indicator colour."""
        self.state_label.setText(f"STATE: {text}")
        self._dot.set_state(color or self._palette.accent, pulse=pulse)

    def set_model(self, text: str) -> None:
        self.model_label.setText(f"CORE: [{text}]" if text else "CORE: [—]")

    def set_device(self, text: str) -> None:
        self.device_label.setText(f"DEV: {text}" if text else "DEV: —")

    def set_tokens(self, used: int, budget: int, *, tps: float = 0.0, latency_s: float = 0.0) -> None:
        """Refresh throughput, latency and context figures after a finished turn."""
        if budget:
            self.token_label.setText(f"CONTEXT: 🧠 {used}/{budget} tok")
        if tps > 0:
            self.tps_label.setText(f"SPEED: ⚡ {tps:.1f} t/s")
        if latency_s > 0:
            self.latency_label.setText(f"LATENCY: ⏱️ {latency_s:.1f}s")

    def reset_metrics(self) -> None:
        """Clear the last turn's figures so a slow request cannot linger."""
        self.tps_label.setText("SPEED: ⚡ -- t/s")
        self.latency_label.setText("LATENCY: ⏱️ --s")


# --------------------------------------------------------------------------- #
# Device picker
# --------------------------------------------------------------------------- #

class DevicePicker(QWidget):
    """A labelled device selector.

    Enumeration is delegated: the caller passes in descriptors, so a missing
    ``sounddevice`` cannot break widget construction.
    """

    def __init__(
        self,
        palette: Palette,
        *,
        label: str,
        placeholder: str = "system default",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._devices: list = []

        self._caption = QLabel(label.upper())
        self._caption.setProperty("tech", True)

        self.combo = QComboBox(self)
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo.addItem(placeholder, None)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._caption, 0)
        layout.addWidget(self.combo, 1)

    @property
    def selection(self) -> int | str | None:
        """The resolved device: an index, a name substring, or None for default."""
        return self.combo.currentData()

    @property
    def devices(self) -> list:
        """The descriptors currently offered, after the input/output filter."""
        return list(self._devices)

    @property
    def label_text(self) -> str:
        return self.combo.currentText()

    def set_devices(self, devices, *, input_only: bool = False, output_only: bool = False) -> None:
        """Replace the options from a list of AudioDevice descriptors."""
        current = self.selection
        self._devices = []
        self.combo.clear()
        self.combo.addItem(self._placeholder, None)
        for device in devices:
            if input_only and not device.is_input:
                continue
            if output_only and not device.is_output:
                continue
            self.combo.addItem(f"[{device.index}] {device.name}", device.index)
            self._devices.append(device)
        if current is not None:
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == current:
                    self.combo.setCurrentIndex(i)
                    break
