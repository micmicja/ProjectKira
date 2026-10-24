"""Visual design tokens and the acrylic HUD look.

There is no reliable way to get real Windows DWM acrylic from PyQt6 alone, so "acrylic"
here is the standard approximation: a translucent frameless window with a rounded inner
panel, a soft vertical gradient, a one-pixel inner highlight, and a drop shadow. Over a
dark wallpaper this reads as glass.

This is the Stark/J.A.R.V.I.S. holographic variant: deep abyss navy on hologram cyan,
monospace throughout, angular surfaces with corner brackets, and amber/red reserved for
states that need attention.

Everything downstream uses the tokens here rather than literal colours, so a theme
change is a one-file edit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QLinearGradient,
    QPainter,
    QPalette,
    QPainterPath,
    QPen,
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
    border: str         # The cyan hairline. Alpha is applied at use sites.
    border_soft: str    # Outer faint stroke.
    glass: str          # Translucent card surface, before alpha.
    glass_2: str        # Second glass tone, for nested panels inside a panel.
    text: str
    text_dim: str
    text_faint: str
    accent: str
    accent_bright: str  # Hover/focus lift of the accent.
    accent_2: str       # Deep arc blue: rings, dividers, secondary strokes.
    accent_soft: str
    danger: str
    success: str
    warning: str
    shadow: str


# Stark holographic cyan. Colours are hex rgb; every translucent use site multiplies
# alpha in on purpose, so the tokens stay legible as plain strings in QSS.
DARK = Palette(
    name="stark-cyan",
    window_bg="#050811",
    panel_a="#0a1526",
    panel_b="#04090f",
    border="#00e5ff",
    border_soft="#0e2c40",
    glass="#071426",
    glass_2="#04090f",
    text="#e0f2fe",
    text_dim="#7dd3fc",
    text_faint="#94a3b8",
    accent="#00e5ff",
    accent_bright="#38bdf8",
    accent_2="#0284c7",
    accent_soft="#0369a1",
    danger="#ef4444",
    success="#38bdf8",
    warning="#f59e0b",
    shadow="#004b5f",
)

LIGHT = Palette(
    name="paper",
    window_bg="#e8ecf3",
    panel_a="#fbfcfe",
    panel_b="#eef1f7",
    border="#2f6bff",
    border_soft="#d3dae6",
    glass="#f4f8fe",
    glass_2="#eef3fb",
    text="#131a26",
    text_dim="#3d6fa8",
    text_faint="#8d98ad",
    accent="#2f6bff",
    accent_bright="#5b8dff",
    accent_2="#2f6bff",
    accent_soft="#dbe6ff",
    danger="#d64545",
    success="#1f9d63",
    warning="#c2700a",
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


# UI_MONOSPACE is the *resolved* family of MONOSPACE, so QSS can name a single
# installed font instead of a comma list Qt would not know how to fall back through.
UI_MONOSPACE = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
UI_MONOSPACE = UI_MONOSPACE or "Courier New"


def MONO_UI_FONT(size: int = 10) -> QFont:
    """A monospace face resolved against installed fonts, for QSS-heavy text."""
    font = QFont(UI_MONOSPACE, size)
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
QMainWindow { background-color: #050811; }
#central_widget { background-color: #050811; border: 1px solid rgba(0, 229, 255, 0.4); }

QWidget { color: $text; font-family: $mono_ui; font-size: 10pt; }
QLabel { background: transparent; }
QLabel[dim="true"] { color: $dim; }
QLabel[faint="true"] { color: $faint; }
QLabel[accent="true"] { color: $accent; font-weight: 600; }
QLabel[accent2="true"] { color: $accent_2; }
QLabel[tech="true"] { color: $dim; font-family: $mono_ui; font-size: 8pt; letter-spacing: 0.3px; }
QLabel[telemetry="true"] { color: $accent; font-family: $mono_ui; font-size: 8pt; letter-spacing: 0.5px; }
QLabel[telemetry="true"][bright="true"] { color: $text; }
QLabel[caption="true"] { color: $accent; font-family: $mono_ui; font-size: 8pt; letter-spacing: 0.7px; }
QLabel[title="true"] { color: $text; font-size: 16pt; font-weight: 700; letter-spacing: 1.6px; font-family: $sans; }

QPushButton {
    background-color: $button_bg;
    color: $dim;
    border: 1px solid $border_glow;
    border-radius: 4px;
    padding: 6px 12px;
    font-family: $mono_ui;
}
QPushButton:hover { background-color: $button_hover; color: $text; border-color: $border_glow_focus; }
QPushButton:pressed { background-color: $button_pressed; border-color: $border_glow_focus; }
QPushButton:disabled { color: $faint; border-color: $border_soft; background-color: transparent; }
QPushButton:focus { border-color: $border_glow_focus; }

QPushButton[pill="true"] { border-radius: 13px; padding: 5px 13px; }
QPushButton[icon="true"] { padding: 6px; border-radius: 8px; }
QPushButton[primary="true"] {
    background-color: $accent; color: #02101a; font-weight: 700;
    border: 1px solid $border_glow_focus;
}
QPushButton[primary="true"]:hover { background-color: $accent_bright; color: #02101a; }
QPushButton[primary="true"]:disabled { color: #1d3a4a; background-color: $border_soft; border-color: $border_soft; }
QPushButton[secondary="true"] {
    background-color: $glass; color: $accent; border: 1px solid $border_glow;
}
QPushButton[secondary="true"]:hover { background-color: $button_hover; border-color: $border_glow_focus; }
QPushButton[danger="true"] {
    background-color: $danger_soft; color: $danger; border: 1px solid $danger_line;
}
QPushButton[danger="true"]:hover { background-color: $danger_soft2; border-color: $danger; }

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: $glass_2;
    border: 1px solid $border_glow;
    border-radius: 4px;
    padding: 8px 10px;
    selection-background-color: $accent;
    selection-color: #02101a;
    font-family: $mono_ui;
}
QLineEdit:focus, QTextEdit:focus { border-color: $border_glow_focus; }

QComboBox {
    background-color: $glass;
    color: $dim;
    border: 1px solid $border_glow;
    border-radius: 4px;
    padding: 4px 8px;
}
QComboBox:hover { border-color: $border_glow_focus; }
QComboBox::drop-down { border: none; width: 20px; color: $accent; }
QComboBox QAbstractItemView {
    background-color: $panel_a;
    border: 1px solid $border_glow;
    border-radius: 4px;
    selection-background-color: $accent;
    selection-color: #02101a;
}

QProgressBar {
    border: 1px solid rgba(0, 229, 255, 0.2);
    background-color: #071426;
    border-radius: 2px;
    height: 8px;
    text-align: right;
}
QProgressBar::chunk {
    background-color: #f59e0b;
    border-radius: 2px;
    margin: 0px;
    width: 1px;
}

QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 4px 0; }
QScrollBar::handle:vertical { background: $scroll; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: $scroll_hover; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; }
QScrollBar::handle:horizontal { background: $scroll; border-radius: 4px; min-width: 24px; }

QToolTip {
    background-color: $glass_solid; color: $text;
    border: 1px solid $border_glow; border-radius: 4px; padding: 6px 8px;
}

QSlider { background: transparent; }
QSlider::groove:horizontal { height: 3px; background: $border_soft; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 13px; height: 13px; margin: -5px 0; border-radius: 7px;
    background: $accent; border: 1px solid $border_glow_focus;
}
"""


def _with_alpha(base: str, alpha: int) -> QColor:
    colour = QColor(base)
    colour.setAlpha(alpha)
    return colour


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
        "accent_bright": palette.accent_bright,
        "accent_2": palette.accent_2,
        "accent_soft": palette.accent_soft,
        "border_soft": palette.border_soft,
        "border_glow": _with_alpha(palette.border, 90).name(QColor.NameFormat.HexArgb),
        "border_glow_focus": _with_alpha(palette.border, 180).name(QColor.NameFormat.HexArgb),
        "danger": palette.danger,
        "danger_soft": _with_alpha(palette.danger, 40).name(QColor.NameFormat.HexArgb),
        "danger_soft2": _with_alpha(palette.danger, 70).name(QColor.NameFormat.HexArgb),
        "danger_line": _with_alpha(palette.danger, 130).name(QColor.NameFormat.HexArgb),
        "glass": _with_alpha(palette.glass, 180).name(QColor.NameFormat.HexArgb),
        "glass_2": _with_alpha(palette.glass_2, 200).name(QColor.NameFormat.HexArgb),
        "glass_solid": _with_alpha(palette.panel_a, 250).name(QColor.NameFormat.HexArgb),
        "divider": _with_alpha(palette.accent, 60).name(QColor.NameFormat.HexArgb),
        "button_bg": _with_alpha(palette.glass, 140).name(QColor.NameFormat.HexArgb),
        "button_hover": _with_alpha(palette.accent, 32).name(QColor.NameFormat.HexArgb),
        "button_pressed": _with_alpha(palette.accent, 54).name(QColor.NameFormat.HexArgb),
        "panel_a": palette.panel_a,
        "sans": QFont(SANS).family(),
        "mono": QFont(MONOSPACE).family(),
        "mono_ui": UI_MONOSPACE,
        "scroll": _with_alpha(palette.accent, 90).name(QColor.NameFormat.HexArgb),
        "scroll_hover": _with_alpha(palette.accent, 160).name(QColor.NameFormat.HexArgb),
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

        # The gradient must span the widget's height in device coordinates; a
        # QLinearGradient(QRectF, ...) is not an overload of this constructor.
        span = float(max(1, self.height()))
        gradient = QLinearGradient(QPointF(0.0, 0.0), QPointF(0.0, span))
        top = QColor(self._palette.panel_a)
        top.setAlpha(232)
        bottom = QColor(self._palette.panel_b)
        bottom.setAlpha(242)
        gradient.setColorAt(0.0, top)
        gradient.setColorAt(1.0, bottom)
        painter.fillPath(path, gradient)

        # Inner highlight on the top edge: the "glass has a surface" cue.
        painter.setPen(QPen(QColor(self._palette.border), 90))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inner = rect.adjusted(0.75, 0.75, -0.75, -0.75)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(inner, self._radius - 0.75, self._radius - 0.75)
        painter.drawPath(inner_path)

        painter.end()


class RoundedCard(QFrame):
    """A transcript/terminal surface.

    Rendered as tech glass -- a translucent navy fill, an accent wash along the top
    edge for the inner-glow cue, and an angular 1px cyan outline with corner brackets.
    """

    def __init__(
        self,
        palette: Palette,
        *,
        parent: QWidget | None = None,
        radius: int = 4,
        tint: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._radius = max(0, radius)
        base = tint if tint else palette.glass
        self._fill = _with_alpha(base, 178)
        self._border = _with_alpha(palette.border, 90)
        self._glow = _with_alpha(palette.border, 40)
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(10)
        effect.setOffset(0, 2)
        effect.setColor(_with_alpha(palette.shadow, 190))
        self.setGraphicsEffect(effect)

    def set_tint(self, color: str, alpha: int = 34) -> None:
        """Retint the surface. Used to lean assistant rows toward the accent."""
        self._fill = _with_alpha(color, alpha)
        self.update()

    def set_radius(self, radius: int) -> None:
        self._radius = max(0, radius)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if rect.width() <= 0 or rect.height() <= 0:
            painter.end()
            return

        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)

        # Tech glass fill, then a soft accent wash so the top edge reads as lit.
        painter.fillPath(path, self._fill)
        glow = QLinearGradient(QPointF(0.0, rect.top()), QPointF(0.0, rect.bottom()))
        glow.setColorAt(0.0, self._glow)
        glow.setColorAt(0.55, _with_alpha(self._palette.border, 0))
        painter.fillPath(path, glow)

        # 1px angular cyan outline.
        pen = QPen(self._border, 1.0)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inner = rect.adjusted(0.5, 0.5, -0.5, -0.5)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(
            inner, max(0.0, self._radius - 0.5), max(0.0, self._radius - 0.5)
        )
        painter.drawPath(inner_path)

        # Tech brackets at the corners: the segmented-outline cue.
        bracket = QColor(self._palette.border)
        bracket.setAlpha(150)
        bpen = QPen(bracket, 1.6)
        bpen.setCapStyle(Qt.PenCapStyle.SquareCap)
        painter.setPen(bpen)
        length = min(13.0, rect.width() * 0.22, rect.height() * 0.22)
        if length >= 3:
            for x, y, sx, sy in (
                (rect.left(), rect.top(), 1, 1),
                (rect.right(), rect.top(), -1, 1),
                (rect.left(), rect.bottom(), 1, -1),
                (rect.right(), rect.bottom(), -1, -1),
            ):
                painter.drawLine(QPointF(x, y), QPointF(x + sx * length, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y + sy * length))
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
        self._size = max(2, size)
        self._t = 0.0
        self._active = False
        self._colour = QColor(palette.text_faint)
        self.setFixedSize(self._size, self._size)
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
        wave = (math.sin(self._t) + 1.0) * 0.5 if self._active else 0.0
        colour = QColor(self._colour)
        colour.setAlpha(int(self._MIN_ALPHA + (self._MAX_ALPHA - self._MIN_ALPHA) * wave))

        # Glow halo behind the disc: the holographic "this is live" cue.
        halo = QColor(self._colour)
        halo.setAlpha(int(36 + 60 * wave))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(halo)
        painter.setPen(Qt.PenStyle.NoPen)
        ring = QRectF(0, 0, self._size, self._size).adjusted(-1.5, -1.5, 1.5, 1.5)
        if ring.width() > 0:
            painter.drawEllipse(ring)

        painter.setBrush(colour)
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
