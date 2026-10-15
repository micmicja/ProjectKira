"""Visual design tokens and the acrylic HUD look.

There is no reliable way to get real Windows DWM acrylic from PyQt6 alone, so "acrylic"
here is the standard approximation: a translucent frameless window with a rounded inner
panel, a soft vertical gradient, a one-pixel inner highlight, and a drop shadow. Over a
dark wallpaper this reads as glass.

Everything downstream uses the tokens here rather than literal colours, so a theme
change is a one-file edit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QEasingCurve, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPalette,
    QPainterPath,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class Palette:
    """Colours for one theme. Dark is the default; light exists for daytime use."""

    name: str
    window_bg: str      # Behind the panel, visible at the rounded corners.
    panel_a: str        # Gradient top of the panel.
    panel_b: str        # Gradient bottom of the panel.
    border: str         # Inner highlight stroke.
    border_soft: str    # Outer faint stroke.
    text: str
    text_dim: str
    text_faint: str
    accent: str
    accent_soft: str
    danger: str
    success: str
    shadow: str


DARK = Palette(
    name="midnight",
    window_bg="#05070d",
    panel_a="#141b29",
    panel_b="#0b1018",
    border="#3a4c8f",
    border_soft="#1b2231",
    text="#eef2f9",
    text_dim="#9aa7bd",
    text_faint="#5f6b82",
    accent="#6ea8ff",
    accent_soft="#22344f",
    danger="#ff6b6b",
    success="#5fd39a",
    shadow="#000000",
)

LIGHT = Palette(
    name="paper",
    window_bg="#e8ecf3",
    panel_a="#fbfcfe",
    panel_b="#eef1f7",
    border="#ffffff",
    border_soft="#d3dae6",
    text="#131a26",
    text_dim="#5b6880",
    text_faint="#8d98ad",
    accent="#2f6bff",
    accent_soft="#dbe6ff",
    danger="#d64545",
    success="#1f9d63",
    shadow="#26314a",
)


def theme(name: str = "dark") -> Palette:
    return DARK if name.lower().startswith("d") else LIGHT


# --------------------------------------------------------------------------- #
# Typography
# --------------------------------------------------------------------------- #

MONOSPACE = "Cascadia Mono, JetBrains Mono, Consolas, Courier New, monospace"
SANS = "Segoe UI Variable, Segoe UI, Inter, system-ui, sans-serif"


def base_font() -> QFont:
    font = QFont(SANS, 10)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.2)
    return font


def mono_font(size: int = 10) -> QFont:
    font = QFont(MONOSPACE, size)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    return font


def ui_font(size: int, *, weight: int = QFont.Weight.Normal, italic: bool = False) -> QFont:
    font = QFont(SANS, size)
    font.setWeight(weight)
    font.setItalic(italic)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.2)
    return font


# --------------------------------------------------------------------------- #
# Application palette and stylesheet
# --------------------------------------------------------------------------- #

def apply_palette(app: QApplication, palette: Palette) -> None:
    """Apply the tokens to the Qt palette so native dialogs match the HUD too."""
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(palette.window_bg))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(palette.text))
    pal.setColor(QPalette.ColorRole.Base, QColor(palette.panel_b))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(palette.panel_a))
    pal.setColor(QPalette.ColorRole.Text, QColor(palette.text))
    pal.setColor(QPalette.ColorRole.Button, QColor(palette.panel_a))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(palette.text))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(palette.accent))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(palette.panel_a))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(palette.text))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(palette.text_faint))
    app.setPalette(pal)


_QSS_TEMPLATE = """
QWidget { color: $text; font-family: $sans; font-size: 10pt; }
QLabel { background: transparent; }
QLabel[dim="true"] { color: $dim; }
QLabel[faint="true"] { color: $faint; }
QLabel[accent="true"] { color: $accent; font-weight: 600; }

QPushButton {
    background-color: $accent_soft;
    color: $text;
    border: 1px solid $border_soft;
    border-radius: 8px;
    padding: 6px 12px;
}
QPushButton:hover { background-color: $accent_hover; border-color: $accent; }
QPushButton:pressed { background-color: $accent_pressed; }
QPushButton:disabled { color: $faint; border-color: $border_soft; }

QPushButton[pill="true"] { border-radius: 14px; padding: 5px 12px; }
QPushButton[primary="true"] {
    background-color: $accent; color: #0a0f18; font-weight: 600; border: none;
}
QPushButton[primary="true"]:hover { background-color: $accent_hover; }
QPushButton[danger="true"] { background-color: transparent; color: $danger; border-color: $danger; }
QPushButton[danger="true"]:hover { background-color: $danger_soft; }
QPushButton[icon="true"] { padding: 6px; border-radius: 10px; }

QTextEdit, QPlainTextEdit, QLineEdit {
    background-color: $field_bg;
    border: 1px solid $border_soft;
    border-radius: 10px;
    padding: 8px 10px;
    selection-background-color: $accent;
    selection-color: #ffffff;
}
QTextEdit:focus, QLineEdit:focus { border-color: $accent; }

QScrollBar:vertical { background: transparent; width: 8px; margin: 4px 0; }
QScrollBar::handle:vertical { background: $scroll; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: $scroll_hover; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; }
QScrollBar::handle:horizontal { background: $scroll; border-radius: 4px; min-width: 24px; }

QToolTip {
    background-color: $panel_a; color: $text;
    border: 1px solid $border_soft; border-radius: 6px; padding: 6px 8px;
}

QComboBox {
    background-color: $field_bg;
    border: 1px solid $border_soft;
    border-radius: 8px;
    padding: 4px 8px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: $panel_a;
    border: 1px solid $border_soft;
    border-radius: 8px;
    selection-background-color: $accent;
    selection-color: #ffffff;
}

QSlider { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: $border_soft; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: $accent;
}
"""


def _with_alpha(base: str, alpha: int) -> str:
    colour = QColor(base)
    colour.setAlpha(alpha)
    return colour.name()


def _lighter(base: str, factor: int) -> str:
    return QColor(base).lighter(factor).name()


def _darker(base: str, factor: int) -> str:
    return QColor(base).darker(factor).name()


def stylesheet(palette: Palette) -> str:
    """The application stylesheet, built from the tokens. One source of truth.

    Tokens are ``$name`` rather than ``{name}`` so the QSS's own braces need no
    escaping, which would otherwise double every rule body.
    """
    import re

    values = {
        "text": palette.text,
        "dim": palette.text_dim,
        "faint": palette.text_faint,
        "accent": palette.accent,
        "accent_soft": palette.accent_soft,
        "accent_hover": _lighter(palette.accent_soft, 128),
        "accent_pressed": _darker(palette.accent_soft, 116),
        "border_soft": palette.border_soft,
        "danger": palette.danger,
        "danger_soft": _with_alpha(palette.danger, 45),
        "field_bg": _darker(palette.panel_b, 108),
        "panel_a": palette.panel_a,
        "sans": QFont(SANS).family(),
        "scroll": _with_alpha(palette.text_faint, 120),
        "scroll_hover": _with_alpha(palette.text_faint, 200),
    }

    def _sub(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"unknown stylesheet token ${key}")
        return values[key]

    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", _sub, _QSS_TEMPLATE)


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #

class AcrylicPanel(QFrame):
    """A rounded translucent panel with a gradient and a soft drop shadow.

    Subclasses paint inside :meth:`content_rect`, which accounts for the panel's own
    padding so children never overhang the glass edge.
    """

    def __init__(
        self,
        palette: Palette,
        *,
        parent: QWidget | None = None,
        radius: int = 20,
        shadow: bool = True,
        transparent_window: bool = False,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._radius = radius
        if shadow:
            effect = QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(48)
            effect.setOffset(0, 10)
            # palette.shadow is already a hex string like "#000000"
            effect.setColor(QColor(palette.shadow))
            self.setGraphicsEffect(effect)
        if transparent_window:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def palette(self) -> Palette:
        return self._palette

    def set_radius(self, radius: int) -> None:
        self._radius = radius
        self.update()

    def content_rect(self, padding: int = 18) -> QRectF:
        """The inner drawing area, inset by the panel's own padding."""
        return QRectF(padding, padding, self.width() - padding * 2, self.height() - padding * 2)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)

        gradient = QLinearGradient(QRectF(0, 0, 0, max(1.0, self.height())))
        gradient.setStart(0, 0)
        gradient.setFinalStop(0, 1)
        top = QColor(self._palette.panel_a)
        top.setAlpha(232)
        bottom = QColor(self._palette.panel_b)
        bottom.setAlpha(242)
        gradient.setColorAt(0.0, top)
        gradient.setColorAt(1.0, bottom)
        painter.fillPath(path, gradient)

        # Inner highlight on the top edge: the "glass has a surface" cue.
        painter.setPen(QColor(self._palette.border, 90))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inner = rect.adjusted(0.75, 0.75, -0.75, -0.75)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(inner, self._radius - 0.75, self._radius - 0.75)
        painter.drawPath(inner_path)

        painter.end()


class RoundedCard(QFrame):
    """A secondary surface for grouping: transcript rows, tool events, status cards."""

    def __init__(self, palette: Palette, *, parent: QWidget | None = None, radius: int = 12) -> None:
        super().__init__(parent)
        self._palette = palette
        self._radius = radius
        self._tint = QColor(_lighter(palette.panel_a, 112))

    def set_tint(self, color: str, alpha: int = 60) -> None:
        self._tint = _with_alpha(color, alpha)
        self.update()

    def set_radius(self, radius: int) -> None:
        self._radius = radius
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        painter.fillPath(path, self._tint)
        painter.setPen(_with_alpha(self._palette.border_soft, 160))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()


class StatusDot(QWidget):
    """A small indicator that pulses only while something is actually happening.

    The animation timer is created on first activation and deleted on deactivation, so
    an idle assistant consumes no CPU.
    """

    _MIN_ALPHA, _MAX_ALPHA = 90, 235

    def __init__(self, palette: Palette, *, parent: QWidget | None = None, size: int = 10) -> None:
        super().__init__(parent)
        self._palette = palette
        self._size = size
        self._t = 0.0
        self._active = False
        self._colour = QColor(palette.text_faint)
        self.setFixedSize(size, size)
        self._timer = None

    def set_state(self, color: str, *, pulse: bool) -> None:
        """Set the dot's colour and whether it animates."""
        self._colour = QColor(color)
        self._active = pulse
        if pulse and self._timer is None:
            from PyQt6.QtCore import QTimer

            self._timer = QTimer(self)
            self._timer.setInterval(33)
            self._timer.timeout.connect(self._tick)
            self._timer.start()
        elif not pulse and self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        self.update()

    def _tick(self) -> None:
        self._t += 0.075
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(self._colour)
        if self._active:
            wave = (math.sin(self._t) + 1.0) * 0.5
            colour.setAlpha(int(self._MIN_ALPHA + (self._MAX_ALPHA - self._MIN_ALPHA) * wave))
        else:
            colour.setAlpha(200)
        painter.setBrush(colour)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self._size, self._size)
        painter.end()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def ease_out(c: float) -> float:
    """Cubic ease-out, for widget-level interpolation."""
    return 1.0 - (1.0 - c) ** 3


def default_easing_curve() -> QEasingCurve:
    return QEasingCurve.Type.OutCubic
