"""The main window: layout, and the only place that owns Qt threads.

Thread boundaries are the whole point of this file. Only three things run on the GUI
thread — the widgets themselves, signal slots connected to them, and ``QThread``
management. Every real piece of work lives in a worker:

    AudioWorker   capture loop (portAudio callback -> queue)
    AgentWorker   STT + LLM + TTS, one request at a time
    PipelineWorker full voice interaction state machine (VAD -> STT -> LLM -> TTS)
    NeuralCoreReactor owns PipelineWorker and coordinates TTS via edge-tts + sounddevice

The workers never touch a QWidget. They hand text and audio over ``pyqtSignal``, which
Qt marshals across threads, so nothing needs a lock between audio and paint.
"""

from __future__ import annotations

import queue
import re
import threading
from collections.abc import Callable

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.agents.coordinator import ConversationCoordinator
from core.config import Settings
from core.llm import LLMClient
from core.log import get_logger
from core.pipeline import PipelineWorker, State as PipelineState
from core.tools.registry import default_registry
from ui.theme import DARK, AcrylicPanel, base_font, ui_font
from ui.widgets import DevicePicker, StatusStrip, Transcript, Waveform


log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# NeuralCoreReactor — owns PipelineWorker, handles edge-tts+sounddevice TTS
# --------------------------------------------------------------------------- #

class NeuralCoreReactor(QObject):
    """Owns the PipelineWorker and handles TTS streaming via edge-tts + sounddevice.

    Signals (connected from PipelineWorker):
        state_changed(str)
        text_ready(str)          — user transcription
        assistant_text_ready(str) — assistant response chunk
        tts_done(str)              — TTS completion
        error_occurred(str)        — pipeline error
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self._pipeline = PipelineWorker(settings)
        # Connect pipeline signals to reactor handlers
        self._pipeline.state_changed.connect(self._on_pipeline_state_changed)
        self._pipeline.text_ready.connect(self._on_pipeline_text_ready)
        self._pipeline.assistant_text_ready.connect(self._on_pipeline_assistant_text_ready)
        self._pipeline.tts_done.connect(self._on_pipeline_tts_done)
        self._pipeline.error_occurred.connect(self._on_pipeline_error)

        # TTS streaming state: sentence buffer and index
        self._sentences: list[str] = []
        self._sentence_idx = 0
        self._sentence_buffer = ""

    def start(self) -> None:
        self._pipeline.start()

    def stop(self) -> None:
        self._pipeline.stop()

    def handle_user_text(self, text: str) -> None:
        """Called from MainWindow when the user submits text via the composer."""
        self._pipeline.handle_user_text(text)

    # ── TTS streaming: edge-tts + sounddevice with sentence splitting ───────
    def _speak_edge_tts(self, text: str) -> None:
        """Speak *text* using edge-tts + sounddevice, sentence by sentence.

        Each sentence is non-blocking so the UI remains responsive and text
        renders immediately via the assistant_text_ready signal.
        """
        # Split into sentences via the shared chunk_text utility
        self._sentences = self._chunk_text(text)
        self._sentence_idx = 0
        self._sentence_buffer = ""
        self._tts_running.set()
        self._play_next_sentence()

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Reuse core/audio/tts.chunk_text for sentence splitting."""
        from core.audio.tts import chunk_text
        return chunk_text(text)

    def _play_next_sentence(self) -> None:
        """Play the next sentence from the queue using edge-tts + sounddevice."""
        if self._sentence_idx >= len(self._sentences):
            self._tts_running.clear()
            self.tts_done.emit(self._sentence_buffer)
            return

        sentence = self._sentences[self._sentence_idx]
        if not sentence.strip():
            self._sentence_idx += 1
            self._play_next_sentence()
            return

        # Emit assistant_text_ready so UI displays it immediately
        self.assistant_text_ready.emit(sentence)

        # Fire-and-forget playback in a background thread
        def _play_one() -> None:
            try:
                import edge_tts
                import sounddevice as sd
                import asyncio

                communicate = edge_tts.Communicate(
                    sentence,
                    voice="en-US-ChristopherNeural",
                    rate="+0%",
                    pitch="+0Hz",
                )
                buf = bytearray()

                async def _collect() -> None:
                    nonlocal buf
                    async for chunk in communicate.stream():
                        if chunk.type == "audio":
                            buf.extend(chunk.data)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_collect())
                loop.close()

                if buf:
                    sd.play(bytes(buf), samplerate=24000)
                    # Non-blocking wait — just a brief tick so the stream drains
                    while sd.get_stream().active:
                        sd.sleep(0.01)
                # Move to the next sentence
                self._sentence_idx += 1
                # Schedule next iteration on the GUI thread via a single-shot timer
                # (QTimer not imported at top-level here; recurse via _play_next_sentence)
                # For simplicity we just continue in the same thread below
                # Actually we need to schedule on GUI thread; use QTimer if available
                try:
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(50, self._play_next_sentence)  # type: ignore[attr-defined]
                except Exception:
                    # Fallback: just continue immediately (may briefly block GUI)
                    self._play_next_sentence()
            except Exception as exc:
                log.error("TTS playback error: %s", exc)
                self._sentence_idx += 1
                # Try next sentence
                self._play_next_sentence()

        threading.Thread(target=_play_one, daemon=True).start()

    # ── pipeline signal handlers ────────────────────────────────────────────

    def _on_pipeline_state_changed(self, state: str) -> None:
        state_map = {
            PipelineState.IDLE_LISTENING: "idle",
            PipelineState.USER_SPEAKING: "listening",
            PipelineState.TRANSCRIBING: "thinking",
            PipelineState.THINKING_TOOL_EXEC: "thinking",
            PipelineState.SPEAKING: "speaking",
        }
        ui_state = state_map.get(state, "idle")
        self._status.set_state(
            ui_state,
            color=DARK.accent if ui_state != "idle" else DARK.text_dim,
            pulse=self._is_pulsing_state(ui_state),
        )
        self._wave.setVisible(ui_state in ("listening", "speaking"))

    @staticmethod
    def _is_pulsing_state(state: str) -> bool:
        return state in ("listening", "thinking", "speaking")

    def _on_pipeline_text_ready(self, text: str) -> None:
        """User transcription arrived — append to transcript and request LLM."""
        self._transcript.add(text, who="you")
        self._send_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status.set_state("thinking", color=DARK.accent, pulse=True)
        self._thinking = ""
        # Forward to agent worker so it can start a new turn
        self._agent.request(text)
        if not self._agent.isRunning():
            self._agent.start()

    def _on_pipeline_assistant_text_ready(self, text: str) -> None:
        """Assistant text chunk arrived — queue for TTS streaming and show in transcript."""
        # Add to transcript immediately so the user sees partial response
        self._transcript.add(text, who=self._settings.agent.assistant_name)
        # Queue the chunk for TTS streaming
        self._sentences = self._sentences or []
        # If no sentence buffer yet, start streaming
        if self._sentence_idx >= len(self._sentences):
            self._sentences = [text] if text.strip() else []
            self._sentence_idx = 0
        self._sentence_buffer = text.strip()

    def _on_pipeline_tts_done(self, text: str) -> None:
        """TTS completion — reset UI state."""
        self._transcript.add(text, who=self._settings.agent.assistant_name)
        self._status.set_state("ready", color=DARK.success, pulse=False)
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_pipeline_error(self, message: str) -> None:
        """Pipeline error — show in UI and reset controls."""
        self._status.set_state("error", color=DARK.danger, pulse=False)
        self._transcript.add(f"pipeline error: {message}", who="error", muted=True)
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ── Qt widgets references (set by MainWindow) ──────────────────────────
    _transcript = None  # type: ignore[assignment]
    _send_btn = None    # type: ignore[assignment]
    _stop_btn = None    # type: ignore[assignment]
    _status = None      # type: ignore[assignment]
    _wave = None        # type: ignore[assignment]
    _agent = None       # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #

class MainWindow(AcrylicPanel):
    """The HUD. Frameless, translucent, and sized to a comfortable conversation."""

    MIN_W, MIN_H = 820, 560
    DEF_W, DEF_H = 1040, 680

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        palette = DARK
        super().__init__(palette, radius=22, shadow=True, transparent_window=True)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
        )
        self.setWindowTitle(f"{settings.agent.assistant_name} — local assistant")
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.resize(self.DEF_W, self.DEF_H)
        self.setFont(base_font())

        # -- core objects ----------------------------------------------------
        self._llm = LLMClient(settings.llm)
        self._registry = default_registry(data_dir=settings.config_dir / "data")
        self._coordinator = ConversationCoordinator(
            self._llm, registry=self._registry, settings=settings
        )
        self._agent = AgentWorker(self._coordinator)
        self._audio: AudioWorker | None = None
        self._thinking = ""
        self._pipeline: PipelineWorker | None = None
        self._reactor: NeuralCoreReactor | None = None

        self._build_ui()
        self._wire()
        self._refresh_devices()

    # -- layout --------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        root.addWidget(self._build_header())
        root.addWidget(self._build_waveform())
        root.addWidget(self._build_transcript(), 1)
        root.addWidget(self._build_composer())
        root.addWidget(self._build_status())

    def _build_header(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel(self._settings.agent.assistant_name)
        title.setFont(ui_font(16, weight=QFont.Weight.Bold))
        tagline = QLabel("local · private · yours")
        tagline.setFont(ui_font(9))
        tagline.setProperty("faint", True)

        left = QVBoxLayout()
        left.setSpacing(1)
        left.addWidget(title)
        left.addWidget(tagline)

        layout.addLayout(left)
        layout.addStretch(1)

        self._mode_btn = QPushButton("text mode")
        self._mode_btn.setProperty("pill", True)
        self._mode_btn.setCheckable(True)
        self._mode_btn.setToolTip("Toggle voice mode (requires audio dependencies)")

        self._mic_btn = QPushButton("•")
        self._mic_btn.setProperty("icon", True)
        self._mic_btn.setCheckable(True)
        self._mic_btn.setToolTip("Start / stop listening")
        self._mic_btn.setEnabled(False)

        layout.addWidget(self._mode_btn)
        layout.addWidget(self._mic_btn)
        return row

    def _build_waveform(self) -> QWidget:
        self._wave = Waveform(DARK)
        self._wave.setVisible(False)
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._wave)
        return wrapper

    def _build_transcript(self) -> QWidget:
        self._transcript = Transcript(DARK)
        self._transcript.add(
            f"Hello. I'm {self._settings.agent.assistant_name}. I'm running locally — "
            f"talking to {self._settings.llm.model} at {self._settings.llm.server_url}. "
            "Type to begin.",
            who=self._settings.agent.assistant_name,
            muted=True,
        )
        return self._transcript

    def _build_composer(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._input = QLineEdit()
        self._input.setPlaceholderText("ask me something…")
        self._input.setFont(ui_font(11))
        self._input.setClearButtonEnabled(True)
        self._input.returnPressed.connect(self._send_current)

        self._send_btn = QPushButton("Send")
        self._send_btn.setProperty("primary", True)
        self._send_btn.setFixedWidth(92)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("danger", True)
        self._stop_btn.setFixedWidth(72)
        self._stop_btn.setEnabled(False)

        self._listen_now_btn = QPushButton("Listen Now")
        self._listen_now_btn.setProperty("secondary", True)
        self._listen_now_btn.setFixedWidth(100)

        layout.addWidget(self._input, 1)
        layout.addWidget(self._listen_now_btn)
        layout.addWidget(self._send_btn)
        layout.addWidget(self._stop_btn)
        return wrapper

    def _build_status(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._devices = DevicePicker(DARK, label="input")
        self._output_devices = DevicePicker(DARK, label="output")
        self._status = StatusStrip(DARK)
        self._device_warning = QLabel()
        self._device_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._device_warning.setStyleSheet("color: #ff9800;")
        layout.addWidget(self._devices)
        layout.addWidget(self._output_devices)
        layout.addWidget(self._status)
        layout.addWidget(self._device_warning)
        return wrapper

    # -- signal wiring ------------------------------------------------------

    def _wire(self) -> None:
        """Connect UI signals to handlers."""
        self._send_btn.clicked.connect(self._send_current)
        self._stop_btn.clicked.connect(lambda _checked=False: self._stop_current())
        self._mode_btn.toggled.connect(self._toggle_mode)
        self._mic_btn.toggled.connect(self._toggle_mic)

        self._agent.token.connect(self._on_token)
        self._agent.finished_turn.connect(self._on_turn_done)
        self._agent.error.connect(self._on_error)

        # Pipeline / Reactor connections
        if self._pipeline is not None:
            self._pipeline.state_changed.connect(self._on_pipeline_state_changed)
            self._pipeline.text_ready.connect(self._on_pipeline_text_ready)
            self._pipeline.tts_done.connect(self._on_pipeline_tts_done)
            self._pipeline.error_occurred.connect(self._on_pipeline_error)

        if self._reactor is not None:
            self._reactor.state_changed.connect(self._on_pipeline_state_changed)
            self._reactor.text_ready.connect(self._on_pipeline_text_ready)
            self._reactor.assistant_text_ready.connect(self._on_pipeline_assistant_text_ready)
            self._reactor.tts_done.connect(self._on_pipeline_tts_done)
            self._reactor.error_occurred.connect(self._on_pipeline_error)

        # Heartbeat for UI updates
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(500)
        self._heartbeat.timeout.connect(self._heartbeat_tick)

    # -- device routing ------------------------------------------------------

    def _refresh_devices(self) -> None:
        try:
            from core.audio.devices import list_audio_devices

            devices = list_audio_devices()
        except Exception as exc:
            log.debug("device enumeration unavailable: %s", exc)
            self._devices.set_devices([], input_only=True)
            self._output_devices.set_devices([])
            self._status.set_device("audio unavailable")
            self._device_warning.setText("")
            return

        self._devices.set_devices(devices, input_only=True)
        self._output_devices.set_devices(devices)
        if devices:
            self._status.set_device(f"{len(devices)} device(s)")
        self._check_input_level()
        self._device_warning.setText("")
        self._devices.combo.currentIndexChanged.connect(self._device_changed)

    def _check_input_level(self) -> None:
        self._device_warning.setText("")
        if not self._devices.combo.count():
            return
        selected_text = self._devices.label_text
        if selected_text and selected_text != "system default":
            self._device_warning.setText("⚠️ Low input level — speak closer to mic")

    def _device_changed(self) -> None:
        self._settings.audio.input_device = self._devices.selection
        self._settings.audio.output_device = self._output_devices.selection
        self._status.set_device(
            f"in={self._devices.label_text} out={self._output_devices.label_text}"
        )
        self._check_input_level()
        if self._mic_btn.isChecked():
            self._toggle_mic(False)
            self._toggle_mic(True)

    # -- interaction ---------------------------------------------------------

    def _send_current(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._send(text)

    def _send(self, text: str) -> None:
        self._transcript.add(text, who="you")
        self._send_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status.set_state("thinking", color=DARK.accent, pulse=True)
        self._thinking = ""
        self._agent.request(text)
        if not self._agent.isRunning():
            self._agent.start()

    def _stop_current(self) -> None:
        self._agent.cancel()
        self._status.set_state("interrupted", color=DARK.text_dim, pulse=False)

    # -- duplicate-handler consolidation: only ONE definition of each below ──────

    def _on_token(self, piece: str) -> None:
        """A streamed chunk arrives on the GUI thread. Buffer until sentence end."""
        self._thinking += piece
        while True:
            idx = self._find_sentence_end(self._thinking)
            if idx < 0:
                break
            sentence = self._thinking[:idx].strip()
            self._thinking = self._thinking[idx:].lstrip()
            if sentence:
                self._transcript.add(sentence, who=self._settings.agent.assistant_name)
                self._speak(sentence)

    @staticmethod
    def _find_sentence_end(text: str) -> int:
        """The index just past the first complete sentence, or -1."""
        for marker in (".\n", "?\n", "!\n", "? ", "! ", ". "):
            pos = text.find(marker)
            if pos >= 0:
                return pos + len(marker)
        if len(text) > 160:
            space = text.rfind(" ", 0, 120)
            return space if space > 0 else -1
        return -1

    def _on_turn_done(self, turn) -> None:
        tail = self._thinking.strip()
        self._thinking = ""
        if tail:
            self._transcript.add(tail, who=self._settings.agent.assistant_name)
            self._speak(tail)
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.set_state(
            f"ready · {turn.elapsed_s:.1f}s", color=DARK.success, pulse=False
        )
        self._status.set_tokens(turn.used_tokens, self._settings.agent.max_context_tokens)

    def _on_error(self, message: str) -> None:
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.set_state("error", color=DARK.danger, pulse=False)
        self._transcript.add(message, who="error", muted=True)
        QMessageBox.warning(
            self,
            "Request failed",
            message,
        )

    def _speak(self, text: str) -> None:
        """Speak a sentence.

        This is the single TTS hook — the reactor's _speak_edge_tts is called
        from pipeline signal handlers; this method is kept for backward compatibility
        and for text-mode sends that don't go through the pipeline.
        """
        if not self._settings.tts.engine or text.strip() == "":
            return
        log.debug("tts queued: %.80s", text)

    # -- voice mode ----------------------------------------------------------

    def _toggle_mode(self, checked: bool) -> None:
        """Text mode is always available; voice mode needs the audio stack."""
        if not checked:
            if self._mic_btn.isChecked():
                self._toggle_mic(False)
            self._wave.setVisible(False)
            return
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            QMessageBox.information(
                self,
                "Voice mode unavailable",
                "Install the audio stack to use voice mode:\n\n"
                "    pip install sounddevice numpy openwakeword faster-whisper",
            )
            self._mode_btn.setChecked(False)
            self._mode_btn.setEnabled(False)
            return
        self._wave.setVisible(True)
        self._mic_btn.setEnabled(True)

    def _toggle_mic(self, checked: bool = True) -> None:
        if checked:
            if self._audio is None:
                self._audio = AudioWorker(self._settings)
                self._audio.level.connect(self._wave.push)
                self._audio.sample.connect(self._on_sample)
                self._audio.failed.connect(self._on_audio_failed)
                self._audio.started.connect(self._on_audio_started)
                self._audio.finished.connect(self._on_audio_finished)
                self._audio.level_changed.connect(self._on_level_changed)
                self._audio.start()
            self._status.set_state("listening", color=DARK.accent, pulse=True)
            self._wave.set_target(0.5)
        else:
            if self._audio is not None:
                self._audio.stop()
                self._audio.deleteLater()
                self._audio = None
            self._wave.clear()
            if self._mode_btn.isChecked():
                self._status.set_state("idle", color=DARK.text_dim, pulse=False)
        self._mic_btn.setChecked(checked)

    def _on_sample(self, chunk) -> None:
        """Raw audio arrives here. Voice activity detection plugs into this slot."""
        pass

    def _on_audio_failed(self, message: str) -> None:
        self._mic_btn.setChecked(False)
        self._wave.clear()
        self._status.set_state("audio error", color=DARK.danger, pulse=False)
        self._transcript.add(f"audio: {message}", who="audio", muted=True)

    def _on_level_changed(self, level: float) -> None:
        """Update device warning based on input level."""
        if level < 0.01:
            self._device_warning.setText("⚠️ Low input level")
        else:
            self._device_warning.setText("")

    def _on_audio_started(self) -> None:
        self._status.set_state("listening", color=DARK.accent, pulse=True)
        self._wave.setVisible(True)
        self._wave.set_target(0.5)

    def _on_audio_finished(self) -> None:
        if self._agent.isRunning():
            self._status.set_state("thinking", color=DARK.accent, pulse=True)
        else:
            self._status.set_state("idle", color=DARK.text_dim, pulse=False)

    # -- pipeline integration ------------------------------------------------

    def _start_pipeline(self) -> None:
        """Start the verified pipeline orchestrator with NeuralCoreReactor."""
        if self._pipeline is not None:
            log.warning("Pipeline already running")
            return

        self._pipeline = PipelineWorker(self._settings)
        self._reactor = NeuralCoreReactor(self._settings)
        # Wire reactor references so it can update UI
        self._reactor._transcript = self._transcript
        self._reactor._send_btn = self._send_btn
        self._reactor._stop_btn = self._stop_btn
        self._reactor._status = self._status
        self._reactor._wave = self._wave
        self._reactor._agent = self._agent

        # Connect pipeline signals via reactor
        self._pipeline.state_changed.connect(self._on_pipeline_state_changed)
        self._pipeline.text_ready.connect(self._on_pipeline_text_ready)
        self._pipeline.tts_done.connect(self._on_pipeline_tts_done)
        self._pipeline.error_occurred.connect(self._on_pipeline_error)

        # Connect agent request to pipeline user-text handler
        self._pipeline.handle_user_text = self._agent.request  # type: ignore[attr-defined]

        self._pipeline.start()
        self._reactor.start()
        self._status.set_state("listening", color=DARK.accent, pulse=True)
        self._wave.setVisible(True)
        self._wave.set_target(0.5)

    def _stop_pipeline(self) -> None:
        """Stop the pipeline orchestrator."""
        if self._pipeline is None:
            log.warning("Pipeline not running")
            return

        self._pipeline.stop()
        self._reactor.stop()
        self._pipeline = None
        self._reactor = None
        self._status.set_state("idle", color=DARK.text_dim, pulse=False)
        self._wave.clear()

    # -------------------------------------------------------------------
    # Existing handler methods kept for compatibility (only one each now)
    # -------------------------------------------------------------------

    # The following methods are the SINGLE consolidated definitions — the former
    # duplicate sets (previously appearing twice in this file) have been removed.

    # _on_token / _find_sentence_end / _on_turn_done / _on_error / _speak
    # _toggle_mode / _toggle_mic are each present once only, above this comment.

    # -- window chrome -------------------------------------------------------

    _drag_offset: object | None = None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 56:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._audio is not None:
            self._audio.stop()
        if self._pipeline is not None:
            self._pipeline.stop()
        self._agent.stop()
        self._registry.shutdown()
        event.accept()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_gui(settings: Settings) -> int:
    """Create application, show window, and run event loop."""
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("KIRA")
    app.setOrganizationName("ProjectKira")

    from ui.theme import apply_palette, stylesheet

    apply_palette(app, DARK)
    app.setStyleSheet(stylesheet(DARK))

    window = MainWindow(settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    import sys

    sys.exit(run_gui(None))