"""End-to-end voice pipeline orchestrator.

Manages the state machine for the complete voice interaction loop:
IDLE_LISTENING -> USER_SPEAKING -> TRANSCRIBING -> THINKING_TOOL_EXEC -> SPEAKING

Handles:
- Audio capture and wake word detection
- Speech activity detection and segmentation
- Speech-to-text transcription
- LLM response generation (streaming)
- Text-to-speech playback (edge-tts + sounddevice, sentence-split)
- Real-time barge-in interruption handling
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from core.audio.capture import AudioCapture
from core.audio.vad import SileroVAD
from core.audio.wakeword import WakeWordDetector, WakeWordResult
from core.audio.stt import SpeechToText, STTResult
from core.llm import LLMClient, ChatMessage
from core.tools.registry import Registry, default_registry
from core.tools.verification import register_verification_tools
from core.log import get_logger
from core.audio.tts import chunk_text
import edge_tts
import sounddevice as sd


log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# State definitions
# --------------------------------------------------------------------------- #

class State:
    IDLE_LISTENING = "idle_listening"
    USER_SPEAKING = "user_speaking"
    TRANSCRIBING = "transcribing"
    THINKING_TOOL_EXEC = "thinking_tool_exec"
    SPEAKING = "speaking"


# --------------------------------------------------------------------------- #
# Pipeline worker QObject with Qt signals
# --------------------------------------------------------------------------- #

class PipelineWorker(QObject):
    """Worker that manages the full voice interaction pipeline.

    Signals the GUI can subscribe to:
        state_changed(str)                — current pipeline state
        text_ready(str)                   — user transcription
        assistant_text_ready(str)         — assistant response (streamed chunks)
        token_progress(str)               — progress token for UI feedback
        tool_result(object)                 — tool execution result
        tts_done(str)                       — TTS completion
        error_occurred(str)                 — pipeline error
    """

    state_changed = pyqtSignal(str)
    text_ready = pyqtSignal(str)        # User transcription
    assistant_text_ready = pyqtSignal(str)  # Assistant response (NEW)
    token_progress = pyqtSignal(str)
    tool_result = pyqtSignal(object)
    tts_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self._llm = LLMClient(settings.llm)
        self._registry = default_registry(data_dir=settings.config_dir)
        register_verification_tools(self._registry)

        self._state = State.IDLE_LISTENING
        self._state_lock = threading.Lock()

        self._audio_queue = queue.Queue()
        self._transcription_queue = queue.Queue()
        self._tts_queue = queue.Queue()
        self._tool_events = queue.Queue()

        self._running = threading.Event()
        self._tts_running = threading.Event()
        self._current_text = ""
        self._current_turn = None
        self._tool_response = None
        self._tts_cancel_requested = False

        # TTS streaming state
        self._sentences: list[str] = []
        self._sentence_idx = 0
        self._sentence_buffer = ""

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        log.info("Starting pipeline...")
        self._set_state(State.IDLE_LISTENING)
        self._running.set()

        # Initialize TTS streaming
        self._tts_running.set()

        # Build the audio capture
        self._audio = AudioCapture(
            device=self.settings.audio.input_device,
            sample_rate=16000,
            blocksize=512,
            channels=1,
        )
        self._audio.start()

        # Start wake word detector
        self._wake_word = WakeWordDetector(
            model_name="hey_kira",
            threshold=0.5,
            cooldown_s=2.0,
        )
        self._wake_word.start()

        # Start VAD
        self._vad = SileroVAD(
            sample_rate=16000,
            threshold=0.5,
            silence_timeout_ms=1200,
        )

        # Start STT worker
        self._start_stt_worker()

        # Start the process loop
        self._process_loop()

    def stop(self) -> None:
        log.info("Stopping pipeline...")
        self._running.clear()
        self._tts_running.clear()
        if self._tts_worker is not None:
            self._tts_worker.stop()
            if self._tts_thread is not None:
                self._tts_thread.join(timeout=5.0)
        if self._stt is not None:
            self._stt.stop()
        if self._stt_thread is not None:
            self._stt_thread.join(timeout=5.0)
        if self._audio is not None:
            self._audio.stop()
        # Stop wake word
        if hasattr(self, "_wake_word"):
            self._wake_word.stop()
        # Reset TTS streaming state
        self._sentences = []
        self._sentence_idx = 0
        self._sentence_buffer = ""
        self._set_state(State.IDLE_LISTENING)

    # ── callbacks from workers ────────────────────────────────────────────

    def _stt_callback(self, result: STTResult) -> None:
        log.info("STT completed: %s", result.text[:100])
        self.text_ready.emit(result.text)
        if self._state == State.USER_SPEAKING:
            self._handle_transcript_result(result.text)

    def _handle_transcript_result(self, text: str) -> None:
        log.info("Received transcription: %s", text)
        self._set_state(State.THINKING_TOOL_EXEC)
        self._execute_tool_or_llm(text)

    # ── LLM dispatch ────────────────────────────────────────────────────────

    def _execute_tool_or_llm(self, text: str) -> None:
        """Send text to the LLM (or a tool if the model calls one) using streaming."""

        def _run_llm():
            try:
                chunks: list[str] = []

                def _sink(piece: str) -> None:
                    chunks.append(piece)
                    # Emit each chunk as a token progress event
                    self.assistant_text_ready.emit(piece)
                    self.token_progress.emit(piece)

                completion = self._llm.stream_chat(
                    [ChatMessage(role="user", content=text)],
                    on_chunk=_sink,
                )
                answer = completion.text.strip()
                # Emit the full answer as assistant_text_ready for the UI
                self.assistant_text_ready.emit(answer)
                # Emit done so TTS finalizes
                self.tts_done.emit(answer)
            except Exception as exc:
                log.error("LLM chat failed: %s", exc)
                self.error_occurred.emit(str(exc))
                self._set_state(State.IDLE_LISTENING)

        threading.Thread(target=_run_llm, daemon=True).start()

    # ── TTS orchestration ──────────────────────────────────────────────────

    def _speak_edge_tts(self, text: str) -> None:
        """Stream TTS using edge-tts + sounddevice with sentence splitting."""
        # Split text into sentences using the shared chunk_text utility
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
            # All sentences done
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

                # Use edge-tts to generate audio bytes
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

                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_collect())
                loop.close()

                if buf:
                    sd.play(bytes(buf), samplerate=24000)
                    # Non-blocking wait — just a brief tick so the stream drains
                    while sd.get_stream().active:
                        sd.sleep(0.01)
                # Move to next sentence
                self._sentence_idx += 1
                # Schedule next sentence on GUI thread via timer (simplified)
                # For simplicity, just continue immediately in this thread
                self._play_next_sentence()
            except Exception as exc:
                log.error("TTS playback error: %s", exc)
                self._sentence_idx += 1
                # Try next sentence
                self._play_next_sentence()

    # ── state management ────────────────────────────────────────────────────

    def _set_state(self, new_state: str) -> None:
        with self._state_lock:
            self._state = new_state
            self.state_changed.emit(new_state)
            log.debug("State changed to %s", new_state)

    # ── process loop ────────────────────────────────────────────────────────

    def _process_loop(self) -> None:
        """Main pipeline loop — runs on the QThread event loop."""

        def _loop() -> None:
            while self._running.is_set():
                try:
                    time.sleep(0.1)

                    # Drain audio queue and process VAD/wake word
                    while not self._audio._queue.empty():
                        chunk = self._audio._queue.get()
                        self._process_audio_chunk(chunk)

                    # Handle barge-in
                    self._handle_barge_in()

                    # VAD-based speech end detection
                    if self._state == State.USER_SPEAKING:
                        if self._speech_started and not self._vad.has_speech():
                            self._process_speech_end()

                except Exception as exc:
                    log.error("Pipeline loop error: %s", exc)
                    self.error_occurred.emit(str(exc))
                    break

        threading.Thread(target=_loop, daemon=True).start()

    def _process_audio_chunk(self, chunk: np.ndarray) -> None:
        """Process a single audio chunk through VAD and wake word."""
        import numpy as np

        if self._state == State.IDLE_LISTENING:
            result = self._wake_word.process(chunk)
            if result.detected:
                log.info("Wake word detected!")
                self._set_state(State.USER_SPEAKING)
                self._speech_started = True
                self._speech_start_time = time.time()
            return

        vad_result = self._vad.process(chunk)
        if vad_result.is_speech:
            self._speech_buffer.append(chunk)
            if not self._speech_started:
                self._speech_started = True
                self._speech_start_time = time.time()
        else:
            if self._speech_started:
                elapsed = time.time() - self._speech_start_time
                if elapsed > self._speech_timeout:
                    self._process_speech_end()

    def _process_speech_end(self) -> None:
        """Called when speech segment ends — queue for transcription."""
        if not self._speech_buffer:
            return
        self._transcription_queue.put((b"".join(self._speech_buffer).tobytes(), None))
        self._speech_buffer = []
        self._speech_started = False
        self._set_state(State.TRANSCRIBING)

    def _handle_barge_in(self) -> None:
        if self._state != State.SPEAKING:
            return

        # Check for wake word in incoming audio
        while not self._audio._queue.empty():
            chunk = self._audio._queue.get_nowait()
            result = self._wake_word.process(chunk)
            if result.detected:
                log.info("Barge-in detected!")
                self._cancel_tts()
                self._set_state(State.USER_SPEAKING)
                return

        # Check VAD for barge-in
        if self._vad.has_speech():
            log.info("VAD barge-in detected!")
            self._cancel_tts()
            self._set_state(State.USER_SPEAKING)
            self._speech_buffer = []
            self._speech_started = False

    def _cancel_tts(self) -> None:
        log.info("Cancelling TTS")
        self._tts_running.clear()
        # sounddevice stream will stop naturally; just reset state

    # ── public API ─────────────────────────────────────────────────────────

    def handle_user_text(self, text: str) -> None:
        """Called from the agent worker when the user submits text."""
        log.info("Pipeline received user text: %s", text[:100])
        self._set_state(State.USER_SPEAKING)
        self._speech_started = True
        self._speech_start_time = time.time()
        self._handle_transcript_result(text)

    # ── existing handlers (kept for compatibility) -------------------------

    def _on_tts_done(self, text: str) -> None:
        log.info("TTS completed: %s", text[:100])
        self.tts_done.emit(text)
        self._set_state(State.IDLE_LISTENING)

    # ------------------------------------------------------------------- #
    # Helper methods (private)
    # ------------------------------------------------------------------- #

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into sentences using the shared chunk_text utility."""
        from core.audio.tts import chunk_text
        return chunk_text(text)


__all__ = ["PipelineWorker", "State"]