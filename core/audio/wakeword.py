"""Wake word detection using openWakeWord.

Wraps the openWakeWord model to process incoming 16kHz mono PCM frames
and emit a signal when the trigger word is detected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.log import get_logger

log = get_logger(__name__)

# Default threshold for "Kira" model from openWakeWord
DEFAULT_THRESHOLD = 0.5
# Default cooldown in seconds after a detection to prevent multi-triggering
DEFAULT_COOLDOWN_S = 2.0


@dataclass(slots=True)
class WakeWordResult:
    """Result of a wake word detection pass."""
    detected: bool
    score: float
    model_name: str


class WakeWordDetector:
    """Real-time wake word detector using openWakeWord.

    Feeds 16kHz mono PCM frames to the model and returns a WakeWordResult
    on each call. Implements a configurable cooldown to prevent
    multiple triggers on a single utterance.
    """

    def __init__(
        self,
        model_name: str = "hey_kira",
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.models_dir = models_dir or Path(os.environ.get("OPENWAKEWORD_MODELS_DIR", ""))
        self._model = None
        self._last_detection_time: float = 0.0
        self._load_model()

    def _load_model(self) -> None:
        """Load the openWakeWord model."""
        try:
            from openwakeword import Model
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord is not installed. Run: pip install openwakeword"
            ) from exc

        # Use ONNX framework for all models (built-in and custom) since
        # tflite-runtime is not available on Windows Python 3.14+
        model_path = self._find_model()
        if model_path:
            log.info("Loading openWakeWord custom model from %s", model_path)
            self._model = Model(
                wakeword_models=[str(model_path)],
                inference_framework="onnx",
            )
        else:
            log.info("Loading openWakeWord built-in model (ONNX): %s", self.model_name)
            self._model = Model(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )

    def _find_model(self) -> Optional[Path]:
        """Find the model file in the models directory or openWakeWord's default locations."""
        if not self.models_dir or not self.models_dir.exists():
            return None
        # Look for .onnx or .tflite files matching the model name
        for ext in (".onnx", ".tflite"):
            candidates = list(self.models_dir.glob(f"*{self.model_name}*{ext}"))
            if candidates:
                return candidates[0]
            # Also try the generic hey_kira pattern
            candidates = list(self.models_dir.glob(f"*hey_kira*{ext}"))
            if candidates:
                return candidates[0]
        return None

    def process(self, audio_chunk: np.ndarray) -> WakeWordResult:
        """Process a chunk of 16 kHz mono int16 audio.

        openWakeWord's streaming buffer is picky about two things, and both bite here:

        * It wants **int16 PCM**, not normalised float. Passing float32 makes its
          ``_buffer_raw_data`` store ``x.tolist()`` as a list of scalars instead of a
          flat list of ints, and the very next call dies with "inhomogeneous shape".
        * It wants a **1-D** array. A ``(samples, 1)`` chunk (the shape sounddevice
          delivers for mono streams) becomes a list of single-element lists, and the
          buffer poisons itself the same way.

        Both are flattened and cast here so the caller can pass whatever the capture
        device hands over without knowing about it.
        """
        if self._model is None:
            raise RuntimeError("Wake word model not loaded")

        if audio_chunk.dtype != np.int16:
            # Tolerate float input by rescaling, but keep the final dtype int16.
            audio_int = np.clip(audio_chunk, -1.0, 1.0) * 32767.0
            audio_int = audio_int.astype(np.int16)
        else:
            audio_int = audio_chunk
        audio_int = np.ascontiguousarray(audio_int.reshape(-1))

        # Get prediction - returns dict like {'hey_mycroft': 0.85}
        prediction = self._model.predict(audio_int)

        # Extract score for our model. The dict key is whatever openWakeWord names the
        # file, which may not match self.model_name (e.g. "hey_kira" vs "hey_kira_v0.1").
        score = prediction.get(self.model_name)
        if score is None:
            score = next(iter(prediction.values()), 0.0)
        score = float(score)
        detected = score >= self.threshold

        # Apply cooldown
        import time
        now = time.monotonic()
        if detected:
            if now - self._last_detection_time < self.cooldown_s:
                detected = False  # Suppress during cooldown
            else:
                self._last_detection_time = now
                log.info("Wake word detected: %s (score=%.3f)", self.model_name, score)

        return WakeWordResult(
            detected=detected,
            score=float(score),
            model_name=self.model_name,
        )

    def reset(self) -> None:
        """Reset the detector state (clear cooldown)."""
        self._last_detection_time = 0.0
        if self._model is not None:
            self._model.reset()