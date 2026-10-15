"""Offline speech-to-text using faster-whisper.

Wraps the Whisper model for streaming transcription of audio segments
produced by the voice activity detector.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from core.log import get_logger

log = get_logger(__name__)

# Supported model sizes and their approximate VRAM usage
MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3")
# Compute types: float16 (GPU), int8 (CPU), float32 (CPU)
COMPUTE_TYPES = ("float16", "int8", "float32")


@dataclass(slots=True)
class STTResult:
    """Result of a transcription pass."""
    text: str
    language: str
    language_probability: float
    duration: float  # audio duration in seconds


class SpeechToText:
    """Offline STT worker using faster-whisper.

    Transcribes audio segments (16kHz mono) in a background thread.
    Clients provide audio via :meth:`transcribe` and receive results
    through a callback.
    """

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        device: str = "cpu",
        *,
        language: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        temperature: float = 0.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
        callback: Optional[Callable[[STTResult], None]] = None,
    ) -> None:
        if model_size not in MODEL_SIZES:
            raise ValueError(f"model_size must be one of {MODEL_SIZES}")
        if compute_type not in COMPUTE_TYPES:
            raise ValueError(f"compute_type must be one of {COMPUTE_TYPES}")
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.patience = patience
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self._callback = callback

        self._model = None
        self._worker_thread: Optional[threading.Thread] = None
        self._request_queue: "queue.Queue[tuple[np.ndarray, object]]" = None
        self._stop_event = threading.Event()
        self._initialized = False

        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc

        log.info(
            "Loading Whisper model %s (%s) on %s",
            self.model_size,
            self.compute_type,
            self.device,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root="~/cache/whisper",
        )
        self._initialized = True

    def _worker(self) -> None:
        """Background worker that processes transcription requests."""
        import queue
        while not self._stop_event.is_set():
            try:
                audio, request_id = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if audio is None:  # Sentinel to stop
                break

            try:
                # Ensure audio is float32 in [-1, 1]
                if audio.dtype != np.float32:
                    # Assume int16 PCM
                    if audio.dtype == np.int16:
                        audio_f32 = audio.astype(np.float32) / 32768.0
                    else:
                        raise ValueError(f"Unsupported audio dtype: {audio.dtype}")
                else:
                    audio_f32 = audio

                # Clip to [-1, 1] just in case
                audio_f32 = np.clip(audio_f32, -1.0, 1.0)

                # Transcribe
                segments, info = self._model.transcribe(
                    audio_f32,
                    language=self.language,
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    patience=self.patience,
                    length_penalty=self.length_penalty,
                    temperature=self.temperature,
                    compression_ratio_threshold=self.compression_ratio_threshold,
                    log_prob_threshold=self.log_prob_threshold,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=self.condition_on_previous_text,
                    vad_filter=True,  # We already did VAD, but extra filtering doesn't hurt
                )

                # Collect text from segments
                text = "".join(segment.text for segment in segments).strip()

                result = STTResult(
                    text=text,
                    language=info.language,
                    language_probability=info.language_probability,
                    duration=len(audio_f32) / 16000,  # approximate
                )

                if self._callback:
                    try:
                        self._callback(result)
                    except Exception as exc:
                        log.warning("STT callback raised: %s", exc)
            except Exception as exc:
                log.error("STT transcription failed: %s", exc)
                # Optionally invoke callback with error? For now just log.
            finally:
                self._request_queue.task_done()

    def start(self) -> None:
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            log.warning("STT worker already started")
            return
        import queue
        self._request_queue = queue.Queue()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker,
            name="stt-worker",
            daemon=True,
        )
        self._worker_thread.start()
        log.info("STT worker started")

    def stop(self) -> None:
        """Stop the background worker thread."""
        if self._worker_thread is None:
            return
        log.info("Stopping STT worker")
        self._stop_event.set()
        # Wake the worker with a sentinel
        if self._request_queue is not None:
            self._request_queue.put((None, None))
        self._worker_thread.join(timeout=5.0)
        if self._worker_thread.is_alive():
            log.warning("STT worker did not stop gracefully")
        self._worker_thread = None
        self._request_queue = None

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        request_id: Optional[object] = None,
    ) -> None:
        """Submit audio for transcription.

        Args:
            audio: numpy array of audio samples (int16 or float32) at 16kHz mono.
            request_id: optional identifier to match request with result (not used
                directly but passed to worker for tracking).
        """
        if not self._initialized:
            raise RuntimeError("STT model not loaded")
        if self._request_queue is None:
            raise RuntimeError("STT worker not started; call start() first")
        self._request_queue.put((audio, request_id))

    def is_running(self) -> bool:
        """Return True if the worker thread is alive."""
        return (
            self._worker_thread is not None
            and self._worker_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def __enter__(self) -> "SpeechToText":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()