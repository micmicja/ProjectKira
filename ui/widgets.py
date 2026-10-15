"""HUD widgets: the waveform, the transcript, and the status strip.

These are deliberately thin views. They hold no pipeline state and take no settings:
the window owns the conversation and pushes text down. That keeps them easy to test
and free of the audio/LLM lifecycle.
"""

from __future__ import annotations

import math
import time

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from ui.theme import (
    Palette,
    RoundedCard,
    StatusDot,
    mono_font,
    ui_font,
)


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
# Transcript
# --------------------------------------------------------------------------- #

def _row(text: str, *, palette: Palette, who: str, muted: bool) -> RoundedCard:
    card = RoundedCard(palette, radius=12)
    layout = QHBoxLayout(card)
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(10)

    is_assistant = who.lower() in {"kira", "assistant", "ai"}
    dot = StatusDot(palette, size=8)
    dot.set_state(
        palette.text_faint if muted else (palette.accent if is_assistant else palette.success),
        pulse=False,
    )

    body = QVBoxLayout()
    body.setSpacing(3)
    caption = QLabel(who)
    caption.setFont(ui_font(8, weight=QFont.Weight.DemiBold))
    caption.setProperty("faint", True)
    message = QLabel(text)
    message.setFont(ui_font(10))
    message.setWordWrap(True)
    if muted:
        message.setProperty("dim", True)
    body.addWidget(caption)
    body.addWidget(message)

    layout.addWidget(dot)
    layout.addLayout(body, 1)
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

    def add(self, text: str, *, who: str = "you", muted: bool = False) -> RoundedCard:
        """Append one exchange. Returns the widget so the caller can animate it."""
        card = _row(text, palette=self._palette, who=who, muted=muted)
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._rows.append(card)
        while len(self._rows) > self._max_rows:
            old = self._rows.pop(0)
            self._layout.removeWidget(old)
            old.deleteLater()
        self._scroll_to_bottom()
        self.line_added.emit()
        return card

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear(self) -> None:
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    @property
    def row_count(self) -> int:
        return len(self._rows)


# --------------------------------------------------------------------------- #
# Status strip
# --------------------------------------------------------------------------- #

class StatusStrip(QWidget):
    """Device routing, connection state, and token budget in one line."""

    def __init__(self, palette: Palette, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._dot = StatusDot(palette, size=9)
        self._dot.set_state(palette.text_faint, pulse=False)

        self.state_label = QLabel("idle")
        self.state_label.setFont(ui_font(9, weight=QFont.Weight.DemiBold))
        self.model_label = QLabel("—")
        self.model_label.setFont(mono_font(8))
        self.model_label.setProperty("dim", True)
        self.device_label = QLabel("device —")
        self.device_label.setFont(mono_font(8))
        self.device_label.setProperty("dim", True)
        self.token_label = QLabel("")
        self.token_label.setFont(mono_font(8))
        self.token_label.setProperty("faint", True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._dot)
        layout.addWidget(self.state_label)
        layout.addWidget(self.model_label)
        layout.addSpacerItem(QSpacerItem(8, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        layout.addWidget(self.device_label)
        layout.addWidget(self.token_label)

    def set_state(self, text: str, *, color: str | None = None, pulse: bool = False) -> None:
        """Update the activity label and its indicator colour."""
        self.state_label.setText(text)
        self._dot.set_state(color or self._palette.accent, pulse=pulse)

    def set_model(self, text: str) -> None:
        self.model_label.setText(text)

    def set_device(self, text: str) -> None:
        self.device_label.setText(text)

    def set_tokens(self, used: int, budget: int) -> None:
        self.token_label.setText(f"{used}/{budget} tok" if budget else "")


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

        self._caption = QLabel(label)
        self._caption.setFont(ui_font(9, weight=QFont.Weight.DemiBold))
        self._caption.setProperty("dim", True)

        self.combo = QComboBox(self)
        self.combo.setFont(ui_font(9))
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
    def label_text(self) -> str:
        return self.combo.currentText()

    def set_devices(self, devices, *, input_only: bool = False, output_only: bool = False) -> None:
        """Replace the options from a list of AudioDevice descriptors."""
        current = self.selection
        self.combo.clear()
        self.combo.addItem(self._placeholder, None)
        for device in devices:
            if input_only and not device.is_input:
                continue
            if output_only and not device.is_output:
                continue
            self.combo.addItem(f"[{device.index}] {device.name}", device.index)
        if current is not None:
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == current:
                    self.combo.setCurrentIndex(i)
                    break
