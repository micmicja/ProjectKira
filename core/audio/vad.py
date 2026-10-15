"""Voice activity detection using Silero VAD.

Wraps silero-vad's ``VADIterator`` for streaming speech detection.

The upstream model has one hard requirement that the rest of the pipeline does not
impose: it wants exactly 512 samples at 16 kHz (or 256 at 8 kHz) per call. Anything
else raises inside TorchScript. So this wrapper owns a small internal buffer and
feeds the model in 512-sample windows regardless of how callers chunk their audio.
The caller stays decoupled from the model's block size; the wrapper is the only
thing that has to know it.

Sliding window: an incoming block of N samples is processed as the blocks
``[0:512], [512:1024], ...,`` with the tail under 512 carried forward. That keeps
inference correct for arbitrary block sizes, at the cost of re-inference on the
overlapping regions -- negligible at 512-sample cadence, and the alternative
(fixed-size ring buffer) is more state for the same coverage.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from core.log import get_logger

log = get_logger(__name__)


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SILENCE_TIMEOUT_MS = 300  # How long to wait before calling a gap silence
DEFAULT_SPECTRAL_PAD_MS = 100     # Pad speech onsets and offsets by this much


@dataclass(slots=True)
class VADResult:
    """Result of one VAD pass on a chunk of audio.

    ``speech_probability`` is the highest speech probability across the model
    windows the chunk produced -- not the last one, so a chunk that begins with a
    burst of speech and ends in silence is still reported as speech.
    """

    speech_probability: float
    is_speech: bool
    frame_position: int  # Sample offset within the current speech segment


class SileroVAD:
    """Streaming Silero VAD wrapper with a strict 512-sample feed discipline."""

    # Frames the underlying model accepts at the default sample rate.
    _FRAME_SAMPLES = {16000: 512, 8000: 256}

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        threshold: float = 0.5,
        *,
        min_silence_duration_ms: int = DEFAULT_SILENCE_TIMEOUT_MS,
        silence_timeout_ms: int | None = None,
        spectral_pad_ms: int = DEFAULT_SPECTRAL_PAD_MS,
    ) -> None:
        if sample_rate not in self._FRAME_SAMPLES:
            raise ValueError(
                f"Silero VAD supports {sorted(self._FRAME_SAMPLES)}, got {sample_rate}"
            )
        self.sample_rate = sample_rate
        self.threshold = float(threshold)
        self._frame = self._FRAME_SAMPLES[sample_rate]
        self.silence_timeout_ms = int(
            silence_timeout_ms if silence_timeout_ms is not None else min_silence_duration_ms
        )
        self.spectral_pad_ms = int(spectral_pad_ms)

        self._model = None
        self._iterator = None
        self._pending = np.empty(0, dtype=np.int16)  # tail carried between calls
        self._sample_count = 0  # samples processed since the last reset
        self._last_speech_frame = -1
        self._speech_buffer: deque[np.ndarray] = deque()
        self._silence_start: float | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the Silero VAD model and create the streaming iterator."""
        try:
            from silero_vad import load_silero_vad, VADIterator
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "silero-vad is not installed. Run:  pip install silero-vad"
            ) from exc

        log.info("Loading Silero VAD model")
        self._model = load_silero_vad()
        self._model.eval()

        # VADIterator owns the speech-start / speech-end state machine; we just read
        # the per-window speech probability it returns.
        self._iterator = VADIterator(
            model=self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.silence_timeout_ms,
            speech_pad_ms=self.spectral_pad_ms,
        )

    # -- public API --------------------------------------------------------------

    def process(self, audio_chunk: np.ndarray) -> VADResult:
        """Advance the VAD over a chunk of mono audio.

        Args:
            audio_chunk: mono audio, 1-D or ``(samples, 1)``. Int16 is preferred;
                float32 in ``[-1, 1]`` is also accepted.

        Returns:
            VADResult describing the whole chunk, not just its tail.
        """
        if self._iterator is None:  # pragma: no cover - defensive
            raise RuntimeError("Silero VAD model not loaded")

        # Flatten first: sounddevice delivers ``(frames, channels)``, and the
        # buffer would otherwise poison itself on the next call.
        samples = np.ascontiguousarray(audio_chunk).reshape(-1)
        if samples.size == 0:
            return VADResult(speech_probability=0.0, is_speech=False,
                             frame_position=self._last_speech_frame)
        if samples.dtype != np.int16:
            if samples.dtype == np.float64:
                samples = samples.astype(np.float32)
            samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
            samples = (samples * 32767.0).astype(np.int16)

        # Carry any tail over from the previous call and slide the rest in blocks.
        self._pending = np.concatenate([self._pending, samples])
        self._sample_count += samples.size
        peak = 0.0
        offset = 0
        while len(self._pending) >= self._frame:
            block = self._pending[:self._frame]
            self._pending = self._pending[self._frame:]
            peak = max(peak, self._run_window(block))
            offset += self._frame

        is_speech = peak >= self.threshold
        if is_speech:
            self._last_speech_frame = self._sample_count
            self._silence_start = None
            self._speech_buffer.append(samples.astype(np.int16, copy=True))
        elif self._silence_start is None:
            # First silent frame after some speech: arm the timeout.
            if self._last_speech_frame >= 0:
                self._silence_start = time.monotonic()

        # End-of-speech: silence has held for the configured duration.
        if self._silence_start is not None:
            silent_for = (time.monotonic() - self._silence_start) * 1000
            if silent_for >= self.silence_timeout_ms:
                # A completed speech segment is signalled by clearing the timer,
                # which lets callers that were reading ``is_speech`` see the drop.
                self._silence_start = None

        return VADResult(
            speech_probability=float(peak),
            is_speech=is_speech,
            frame_position=self._last_speech_frame,
        )

    def _run_window(self, block: np.ndarray) -> float:
        """Run one 512-sample window through the model and return its speech probability."""
        tensor = torch.from_numpy(block.astype(np.float32) / 32768.0)
        with torch.no_grad():
            prob = float(self._model(tensor, self.sample_rate).item())
        # Also drive the upstream state machine; it owns the speech-start/end dict.
        with torch.no_grad():
            self._iterator(tensor, return_seconds=False)
        return prob

    def has_speech(self) -> bool:
        """True if any window since the last reset flagged speech."""
        return self._last_speech_frame >= 0

    def get_speech_segment(self) -> Optional[np.ndarray]:
        """Concatenate every chunk we retained while speech was active."""
        if not self._speech_buffer:
            return None
        return np.concatenate(self._speech_buffer).astype(np.int16)

    def reset(self) -> None:
        """Clear internal state so the next chunk is treated as a fresh segment."""
        self._pending = np.empty(0, dtype=np.int16)
        self._sample_count = 0
        self._last_speech_frame = -1
        self._speech_buffer.clear()
        self._silence_start = None
        if self._iterator is not None:
            self._iterator.reset_states()

    def reset_segment(self) -> None:
        """Keep the VAD's internal state but drop the retained audio buffer.

        Use this between turns without re-warming the model.
        """
        self._speech_buffer.clear()
        self._last_speech_frame = -1
        self._silence_start = None

    def __enter__(self) -> "SileroVAD":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None
