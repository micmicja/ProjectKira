"""End-to-end voice pipeline orchestrator.

Manages the state machine for the complete voice interaction loop:
IDLE_LISTENING -> USER_SPEAKING -> TRANSCRIBING -> THINKING_TOOL_EXEC -> SPEAKING

Handles:
- Audio capture and wake word detection
- Speech activity detection and segmentation
- Speech-to-text transcription
- LLM response generation
- Text-to-speech playback
- Real-time barge-in interruption handling
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from core.audio.capture import AudioCapture
from core.audio.vad import SileroVAD
from core.audio.wakeword import WakeWordDetector, WakeWordResult
from core.audio.stt import SpeechToText
from core.llm import LLMClient
from core.tools.registry import Registry
from core.tools.builtin import default_registry
from core.tools.verification import register_verification_tools
from core.log import get_logger


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
    """Worker that manages the full voice interaction pipeline."""

    state_changed = pyqtSignal(str)
    text_ready = pyqtSignal(str)
    tool_result = pyqtSignal(object)
    tts_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    token_progress = pyqtSignal(str)  # For TTS progress indication

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self._llm = LLMClient(settings.llm)
        self._registry = default_registry(data_dir=settings.config_dir / "data")
        register_verification_tools(self._registry)

        # State management
        self._state = State.IDLE_LISTENING
        self._state_lock = threading.Lock()

        # Communication queues
        self._audio_queue = queue.Queue()  # Raw audio chunks from capture
        self._transcription_queue = queue.Queue()  # (audio_chunk, request_id)
        self._tts_queue = queue.Queue()  # (text, call_id)
        self._tool_events = queue.Queue()  # (tool_name, arguments)

        # Worker flags
        self._running = threading.Event()
        self._tts_running = threading.Event()
        self._current_text = ""
        self._current_turn = None
        self._tool_response = None
        self._tts_cancel_requested = False

        # State machine helpers
        self._speech_buffer = []
        self._speech_start_time = 0.0
        self._speech_timeout = 5.0  # seconds of silence before considering speech ended
        self._speech_timer = None

        # Signals
        self._state_changed = State.IDLE_LISTENING

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def start(self) -> None:
        """Start the pipeline: audio capture, wake word detection, VAD, etc."""
        log.info("Starting pipeline...")
        self._state = State.IDLE_LISTENING
        self._state_changed.emit(self._state)
        self._running.set()

        # Start audio capture
        self._audio = AudioCapture(
            device=self.settings.audio.input_device,
            sample_rate=16000,
            blocksize=512,
            channels=1,
        )
        self._audio.start()

        # Start wake word detection
        self._wake_word = WakeWordDetector(
            model_name="hey_kira",
            threshold=0.5,
            cooldown_s=2.0,
        )

        # Start VAD
        self._vad = SileroVAD(
            sample_rate=16000,
            threshold=0.5,
            silence_timeout_ms=1200,
        )

        # Start STT worker
        self._stt = None
        self._stt_thread = None
        self._start_stt_worker()

        # Start TTS worker (initially stopped)
        self._tts_worker = None
        self._tts_thread = None

        # Start the main loop
        self._process_loop()

    def stop(self) -> None:
        """Stop the pipeline and clean up resources."""
        log.info("Stopping pipeline...")
        self._running.clear()
        self._tts_running.set()
        if self._tts_worker:
            self._tts_worker.stop()
            self._tts_thread.join(timeout=5.0)
        self._stt.stop()
        if self._stt_thread:
            self._stt_thread.join(timeout=5.0)
        if self._audio:
            self._audio.stop()
        self._state = State.IDLE_LISTENING
        self._state_changed.emit(self._state)

    def _start_stt_worker(self) -> None:
        """Start the STT worker thread."""
        self._stt = SpeechToText(
            model_size=self.settings.llm.stt_model_size,
            compute_type=self.settings.llm.stt_compute_type,
            on_result=self._on_stt_result,
        )
        self._stt_thread = threading.Thread(target=self._stt.run, daemon=True)
        self._worker_thread = self._stt_thread
        self._stt_thread.start()
        log.info("STT worker started")

    def _on_stt_result(self, result: "STTResult") -> None:
        """Callback from STT worker when transcription is complete."""
        log.debug("STT result received: %s", result.text[:100])
        self.text_ready.emit(result.text)

    # ------------------------------------------------------------------- #
    # Main loop
    # ------------------------------------------------------------------- #

    def _process_loop(self) -> None:
        """Main loop that processes audio and manages state transitions."""
        while self._running.is_set():
            time.sleep(0.1)  # Small sleep to avoid 100% CPU

            # Check for wake word (non-blocking)
            if self._state in (State.IDLE_LISTENING, State.USER_SPEAKING):
                # Drain audio queue for wake word detection
                while not self._audio._queue.empty():
                    chunk = self._audio._queue.get_nowait()
                    # Check for wake word in this chunk
                    result = self._wake_word.process(chunk)
                    if result.detected:
                        log.info("Wake word detected! (score=%.3f)", result.score)
                        self._state = State.USER_SPEAKING
                        self._state_changed.emit(self._state)
                        self._speech_start_time = time.time()
                        self._speech_timer = time.time()
                        continue

            # Handle VAD and speech detection
            if self._state in (State.IDLE_LISTENING, State.USER_SPEAKING):
                # Process audio through VAD
                try:
                    chunk = self._audio.read(timeout=0.1)
                except queue.Empty:
                    continue

                vad_result = self._vad.process(chunk)
                if vad_result.is_speech:
                    # Speech detected
                    self._speech_buffer.append(chunk)
                    if not self._speech_started:
                        self._speech_started = True
                        self._speech_start_time = time.time()
                        log.debug("Speech started at frame %d", self._frame_count)
                else:
                    # No speech - check timeout
                    if self._speech_started:
                        elapsed = time.time() - self._speech_start_time
                        if elapsed > self._speech_timeout:
                            log.info("Speech timeout after %.1fs", elapsed)
                            self._process_speech_end()

    def _process_speech_end(self) -> None:
        """Handle end of speech: send buffered audio to STT."""
        if not self._speech_buffer:
            return

        # Send buffered audio to STT worker
        self._transcription_queue.put((b"".join(self._speech_buffer).tobytes(), None))
        self._speech_buffer = []
        self._speech_started = False
        log.info("Sent audio to STT for transcription")

    def _start_stt_worker(self) -> None:
        """Initialize and start the STT worker."""
        self._stt = SpeechToText(
            model_size=self.settings.llm.stt_model_size,
            compute_type=self.settings.llm.stt_compute_type,
            callback=self._stt_callback,
        )
        self._stt_thread = threading.Thread(
            target=self._stt.run,
            name="stt-worker",
            daemon=True,
        )
        self._stt_thread.start()
        log.info("STT worker started")

    def _stt_callback(self, result: "STTResult") -> None:
        """Callback from STT worker."""
        log.info("STT completed: %s", result.text[:100])
        self.text_ready.emit(result.text)

    # ------------------------------------------------------------------- #
    # State handling
    # ------------------------------------------------------------------- #

    def _set_state(self, new_state: str) -> None:
        """Change state and emit signal."""
        with self._state_lock:
            self._state = new_state
            self.state_changed.emit(new_state)
            log.debug("State changed to %s", new_state)

    def _set_state_user_speaking(self) -> None:
        self._set_state(State.USER_SPEAKING)

    def _set_state_transcribing(self) -> None:
        self._set_state(State.TRANSCRIBING)

    def _set_state_thinking_tool_exec(self) -> None:
        self._set_state(State.THINKING_TOOL_EXEC)

    def _set_state_speaking(self) -> None:
        self._set_state(State.SPEAKING)

    # ------------------------------------------------------------------- #
    # State transition helpers
    # ------------------------------------------------------------------- #

    def _transition_to_user_speaking(self) -> None:
        self._set_state_user_speaking()
        log.info("Entering USER_SPEAKING state")

    def _set_speech_timer(self) -> None:
        """Start or reset the speech timeout timer."""
        self._speech_start_time = time.time()
        self._speech_timer = time.time()

    def _process_speech_end(self) -> None:
        """Called when speech ends: send buffered audio to STT."""
        if not self._speech_buffer:
            return

        # Send buffered audio to STT worker
        self._transcription_queue.put((b"".join(self._speech_buffer).tobytes(), None))
        log.info("Sent %d frames to STT for transcription", len(self._speech_buffer))
        self._speech_buffer = []
        self._speech_started = False

    def _handle_barge_in(self) -> None:
        """Check for barge-in during SPEAKING state."""
        if self._state != State.SPEAKING:
            return

        # Check if wake word detected during speech
        if self._state == State.SPEAKING:
            # Drain audio queue for wake word detection
            while not self._audio._queue.empty():
                chunk = self._audio._queue.get_nowait()
                result = self._wake_word.process(chunk)
                if result.detected:
                    log.info("Barge-in wake word detected! Cancelling TTS")
                    self._cancel_tts()
                    self._state = State.USER_SPEAKING
                    self._state_changed.emit(self._state)
                    return

            # Check VAD for speech onset during speaking
            if self._vad.has_speech():
                # If VAD detects new speech while already speaking, treat as barge-in
                log.info("VAD detected speech during TTS playback - barge-in")
                self._cancel_tts()
                self._state = State.USER_SPEAKING
                self._state_changed.emit(self._state)
                self._speech_buffer = []
                self._speech_started = False
                self._speech_start_time = 0.0
                self._speech_timer = None

    def _cancel_tts(self) -> None:
        """Cancel ongoing TTS playback."""
        log.info("Cancelling TTS playback")
        self._tts_running.clear()
        if self._tts_worker:
            self._tts_worker.cancel()
            self._tts_worker.stop()

    # ------------------------------------------------------------------- #
    # Audio processing
    # ------------------------------------------------------------------- #

    def _process_audio_chunk(self, chunk: np.ndarray) -> None:
        """Process a single audio chunk based on current state."""
        if self._state == State.IDLE_LISTENING:
            # Just forward to wake word detector
            self._wake_word.process(chunk)
            return

        # For USER_SPEAKING and TRANSCRIBING states, feed audio to VAD
        vad_result = self._vad.process(chunk)
        if vad_result.is_speech:
            self._speech_buffer.append(chunk)
            if not self._speech_started:
                self._speech_started = True
                self._speech_start_time = time.time()
                self._speech_timer = time.time()
        else:
            # No speech - check timeout
            if self._speech_started:
                elapsed = time.time() - self._speech_start_time
                if elapsed > self._speech_timeout:
                    self._process_speech_end()

    def _process_speech_end(self) -> None:
        """Called when speech ends: send buffered audio to STT."""
        if not self._speech_buffer:
            return

        # Send buffered audio to STT worker
        self._transcription_queue.put((b"".join(self._speech_buffer).tobytes(), None))
        log.info("Sent %d frames to STT for transcription", len(self._speech_buffer))
        self._speech_buffer = []
        self._speech_started = False

    # ------------------------------------------------------------------- #
    # STT handling
    # ------------------------------------------------------------------- #

    def _stt_callback(self, result: "STTResult") -> None:
        """Handle STT transcription result."""
        log.info("Transcription result: %s", result.text[:200])
        self.text_ready.emit(result.text)

        # If we're in SPEAKING state and have a result, send to LLM
        if self._state == State.SPEAKING:
            self._handle_transcript_result(result.text)

    def _handle_transcript_result(self, text: str) -> None:
        """Process transcribed text."""
        log.info("Received transcription: %s", text)
        if self._state == State.TRANSCRIBING:
            # In a real system, we might parse commands here
            # For now, just send to LLM for response
            self._state = State.THINKING_TOOL_EXEC
            self.state_changed.emit(self._state)
            log.info("Transitioning to THINKING_TOOL_EXEC")
            self._execute_tool_or_llm(text)

    def _execute_tool_or_llm(self, text: str) -> None:
        """Execute tool or call LLM based on transcribed text."""
        # For simplicity, we'll treat any non-empty text as a request to LLM
        log.info("Sending text to LLM: %s", text)
        self._llm.chat([{"role": "user", "content": text}])
        # In a real system, we'd parse the text to determine if it's a command
        # that requires tool execution, but for now we assume LLM handles it

    # ------------------------------------------------------------------- #
    # TTS handling
    # ------------------------------------------------------------------- #

    def _start_tts_worker(self, text: str) -> None:
        """Start TTS worker to speak the given text."""
        if self._tts_worker is not None and self._tts_worker.isRunning():
            log.warning("TTS already running")
            return

        self._tts_worker = TTSWorker(text, self.settings)
        self._tts_thread = threading.Thread(target=self._tts_worker.run, daemon=True)
        self._tts_thread.start()
        self._tts_running.set()
        log.info("TTS worker started for text: %s", text[:50])

    def _cancel_tts(self) -> None:
        """Cancel ongoing TTS playback."""
        log.info("Cancelling TTS")
        self._tts_running.clear()
        if self._tts_worker:
            self._tts_worker.cancel()
            self._tts_thread.join(timeout=4.0)

    def _on_tts_done(self, text: str) -> None:
        """Called when TTS finishes speaking."""
        log.info("TTS completed: %s", text[:100])
        self.tts_done.emit(text)
        self._state = State.IDLE_LISTENING
        self._state_changed.emit(self._state)

    # ------------------------------------------------------------------- #
    # Barge-in handling
    # ------------------------------------------------------------------- #

    def _check_barge_in(self) -> None:
        """Check for barge-in during SPEAKING state."""
        if self._state != State.SPEAKING:
            return

        # Check wake word detection
        while not self._audio._queue.empty():
            chunk = self._audio._queue.get_nowait()
            result = self._wake_word.process(chunk)
            if result.detected:
                log.info("Barge-in wake word detected during speech")
                self._cancel_tts()
                self._state = State.USER_SPEAKING
                self._state_changed.emit(self._state)
                return

        # Check VAD for new speech onset
        if self._vad.has_speech():
            log.info("VAD detected new speech during TTS - barge-in")
            self._cancel_tts()
            self._state = State.USER_SPEAKING
            self._state_changed.emit(self._state)
            self._speech_buffer = []
            self._speech_started = False
            self._speech_start_time = 0.0
            self._speech_timer = None

    # ------------------------------------------------------------------- #
    # Main loop
    # ------------------------------------------------------------------- #

    def _process_loop(self) -> None:
        """Main loop that processes audio and manages state transitions."""
        while self._running.is_set():
            time.sleep(0.1)

            # Process audio chunks
            while not self._audio._queue.empty():
                chunk = self._audio._queue.get()
                self._process_audio_chunk(chunk)

            # Check for barge-in events
            self._handle_barge_in()

            # Handle state transitions
            if self._state == State.USER_SPEAKING:
                # If we have audio buffered and VAD indicates speech end, transcribe
                if self._speech_started and not self._vad.has_speech():
                    self._process_speech_end()

    # ------------------------------------------------------------------- #
    # Helper methods
    # ------------------------------------------------------------------- #

    def _process_audio_chunk(self, chunk: np.ndarray) -> None:
        """Process audio chunk according to current state."""
        if self._state == State.IDLE_LISTENING:
            # Just forward to wake word detector
            self._wake_word.process(chunk)
            return

        # For USER_SPEAKING and TRANSCRIBING states, feed to VAD
        self._vad.process(chunk)

    def _process_speech_end(self) -> None:
        """Called when speech ends: send buffered audio to STT."""
        if not self._speech_buffer:
            return

        # Send buffered audio to STT worker
        self._transcription_queue.put((b"".join(self._speech_buffer).tobytes(), None))
        log.debug("Sent %d frames to STT", len(self._speech_buffer))
        self._speech_buffer = []
        self._speech_started = False

    def _stt_callback(self, result: "STTResult") -> None:
        """Handle STT transcription completion."""
        log.info("Transcription complete: %s", result.text[:200])
        self.text_ready.emit(result.text)

        # After transcription, we typically want to send to LLM
        if self._state == State.TRANSCRIBING:
            self._state = State.THINKING_TOOL_EXEC
            self.state_changed.emit(self._state)
            log.info("Transitioning to THINKING_TOOL_EXEC")
            self._execute_tool_or_llm(result.text)

    def _execute_tool_or_llm(self, text: str) -> None:
        """Execute tool or call LLM based on transcribed text."""
        # For now, we'll just send to LLM and then TTS
        log.info("Sending to LLM: %s", text)
        # In a real implementation, we might check if this is a command that requires
        # a tool call first, but for simplicity we'll just use LLM
        self._llm.chat([{"role": "user", "content": text}])

    def _on_llm_response(self, response: str) -> None:
        """Handle LLM response."""
        log.info("LLM response received: %s", response[:200])
        self._state = State.SPEAKING
        self._state_changed.emit(self._state)
        self._speak_text(response)

    def _speak_text(self, text: str) -> None:
        """Speak the given text using TTS."""
        log.info("Speaking: %s", text[:100])
        self._speak(text)

    def _speak(self, text: str) -> None:
        """Send text to TTS worker."""
        if self._tts_worker is None:
            self._start_tts_worker(text)
        else:
            # Queue text for TTS
            self._tts_queue.put((text, "tts_request"))

    def _start_tts_worker(self) -> None:
        """Start TTS worker thread."""
        self._tts_worker = TTSWorker(self.settings)
        self._tts_thread = threading.Thread(
            target=self._tts_worker.run, daemon=True
        )
        self._tts_thread.start()
        self._tts_running.set()
        log.info("TTS worker started")

    def _start_tts(self, text: str) -> None:
        """Start TTS for the given text."""
        self._state = State.SPEAKING
        self._state_changed.emit(self._state)
        self._current_text = text
        self._tts_cancel_requested = False
        self._tts_worker.request(text)
        self._tts_running.set()

    def _on_tts_done(self, text: str) -> None:
        """Handle TTS completion."""
        log.info("TTS finished: %s", text[:100])
        self.tts_done.emit(text)
        self._state = State.IDLE_LISTENING
        self._state_changed.emit(self._state)

    def _cancel_tts(self) -> None:
        """Cancel TTS if it's playing."""
        if self._tts_worker and self._tts_running.is_set():
            self._tts_worker.cancel()
            self._tts_running.clear()
            log.info("TTS cancelled due to barge-in")

    # ------------------------------------------------------------------- #
    # Audio processing hook
    # ------------------------------------------------------------------- #

    def _process_audio_chunk(self, chunk: np.ndarray) -> None:
        """Process audio chunk according to current state."""
        if self._state == State.IDLE_LISTENING:
            # Just forward to wake word detector
            self._wake_word.process(chunk)
            return

        # For USER_SPEAKING and TRANSCRIBING states, feed to VAD
        self._vad.process(chunk)

    # ------------------------------------------------------------------- #
    # Helper methods
    # ------------------------------------------------------------------- #

    def _process_speech_end(self) -> None:
        """Handle end of speech: send buffered audio to STT."""
        if not self._speech_buffer:
            return

        # Send buffered audio to STT worker
        self._transcription_queue.put((b"".join(self._speech_buffer).tobytes(), None))
        log.debug("Sent %d frames to STT", len(self._speech_buffer))
        self._speech_buffer = []
        self._speech_started = False

    def _execute_tool_or_llm(self, text: str) -> None:
        """Execute tool or call LLM based on transcribed text."""
        log.info("Processing text: %s", text)
        # For now, just send to LLM
        self._llm.chat([{"role": "user", "content": text}])

    def _on_llm_response(self, response: str) -> None:
        """Handle LLM response."""
        log.info("LLM response: %s", response[:200])
        self._state = State.SPEAKING
        self._state_changed.emit(self._state)
        self._speak_text(response)

    def _speak_text(self, text: str) -> None:
        """Speak text via TTS."""
        log.info("Speaking: %s", text[:100])
        self._start_tts_worker(text)

    def _heartbeat(self) -> None:
        """Periodic health check to ensure components are alive."""
        if not self._running.is_set():
            return
        if self._tts_worker and not self._tts_running.is_set():
            log.warning("TTS worker stopped unexpectedly")
        if self._stt and not self._stt.is_running():
            log.warning("STT worker stopped unexpectedly")
        # Add more health checks as needed


# --------------------------------------------------------------------------- #
# TTS Worker
# ------------------------------------------------------------------- #

class TTSWorker(QObject):
    """Worker that handles text-to-speech synthesis."""

    finished = pyqtSignal(str)  # emitted when TTS completes
    error = pyqtSignal(str)

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self._queue = queue.Queue()
        self._running = threading.Event()
        self._cancel_requested = False

    def request(self, text: str) -> None:
        """Queue text for TTS synthesis."""
        self._queue.put(("speak", text))

    def cancel(self) -> None:
        """Cancel ongoing TTS operation."""
        self._cancel_requested = True
        log.info("TTS cancellation requested")

    def run(self) -> None:
        """Main TTS loop."""
        log.info("TTS worker started")
        self._running.set()
        while self._running.is_set():
            try:
                # Wait for text to speak
                item = self._queue.get(timeout=0.5)
                if item is None:
                    break  # Shutdown signal
                op, text = item
                if op == "speak":
                    if self._cancel_requested:
                        log.info("Skipping TTS due to cancellation")
                        continue
                    log.info("Speaking: %s", text[:100])
                    # Simulate TTS playback (in real implementation, this would be actual TTS)
                    time.sleep(0.1 * len(text))  # Simplified
                    self.finished.emit(text)
                    self._tts_running.clear()
            except Exception as exc:
                log.error("TTS error: %s", exc)
                self.error_occurred.emit(str(exc))
                self._running.clear()
                break

    def cancel(self) -> None:
        """Mark TTS for cancellation."""
        self._cancel_requested = True
        log.info("TTS cancellation requested")


# --------------------------------------------------------------------------- #
# Entry point for GUI integration
# ------------------------------------------------------------------- #

def run_gui(settings) -> int:
    """Create application, show window, and run event loop."""
    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("KIRA")
    app.setOrganizationName("ProjectKira")

    from ui.theme import apply_palette, stylesheet

    apply_palette(app, DARK)
    app.setStyleSheet(stylesheet(DARK))

    window = MainWindow(settings)
    window.show()
    return app.exec()


__all__ = ["PipelineWorker"]