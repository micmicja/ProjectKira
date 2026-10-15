"""Audio input capture manager using sounddevice.

Provides a thread-safe, asynchronous audio input stream that delivers
16kHz mono PCM chunks to downstream consumers via a queue.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from core.audio.devices import list_audio_devices, resolve_device
from core.log import get_logger

log = get_logger(__name__)


class AudioCapture:
    """Manages a sounddevice.InputStream and puts audio chunks into a queue.

    The audio is captured as float32 in the range [-1.0, 1.0] and converted
    to 16-bit PCM (int16) for downstream processing.
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        *,
        sample_rate: int = 16000,
        blocksize: int = 512,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.dtype = dtype

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = threading.Event()
        self._restart_needed = threading.Event()
        self._device_index: Optional[int] = None

        # Resolve the device index once at init; will be re-checked on restart.
        self._resolve_device_index()

    def _resolve_device_index(self) -> None:
        """Resolve the device selection to an index, updating self._device_index."""
        if self.device is None:
            # Use system default input device
            self._device_index = None
            return

        try:
            # If it's already an index, use it; otherwise treat as substring.
            if isinstance(self.device, int) or (isinstance(self.device, str) and self.device.isdigit()):
                index = int(self.device)
                # Verify the index exists and is an input device
                devices = list_audio_devices()
                if any(d.index == index and d.is_input for d in devices):
                    self._device_index = index
                else:
                    raise ValueError(f"Device index {index} is not an input device")
            else:
                # Treat as substring
                index = resolve_device(self.device, inputs_only=True)
                if index is None:
                    raise ValueError(f"No default input device matching {self.device!r}")
                self._device_index = index
        except Exception as exc:
            log.warning("Failed to resolve audio device %r: %s", self.device, exc)
            self._device_index = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Callback from sounddevice.InputStream."""
        if status:
            log.debug("Audio callback status: %s", status)
        if not self._running.is_set():
            return
        # Convert float32 audio to 16-bit PCM
        # indata shape: (frames, channels)
        if indata.dtype != np.float32:
            indata = indata.astype(np.float32)
        # Clip to [-1, 1] and scale to int16 range
        audio_int16 = (np.clip(indata, -1.0, 1.0) * 32767).astype(np.int16)
        # If multichannel, convert to mono by averaging
        if self.channels > 1:
            audio_int16 = audio_int16.mean(axis=1).astype(np.int16)
        # Put the mono chunk into the queue
        self._queue.put_nowait(audio_int16)

    def start(self) -> None:
        """Start the audio input stream."""
        if self._running.is_set():
            log.warning("AudioCapture already started")
            return
        self._resolve_device_index()
        log.info(
            "Starting audio capture (device=%s, rate=%d, blocksize=%d)",
            self._device_index,
            self.sample_rate,
            self.blocksize,
        )
        self._running.set()
        self._restart_needed.clear()
        try:
            self._stream = sd.InputStream(
                device=self._device_index,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            log.error("Failed to start audio capture: %s", exc)
            self._running.clear()
            raise

    def stop(self) -> None:
        """Stop the audio input stream."""
        if not self._running.is_set():
            return
        log.info("Stopping audio capture")
        self._running.clear()
        self._restart_needed.set()  # Wake the wait loop if any
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.warning("Error closing audio stream: %s", exc)
            finally:
                self._stream = None
        # Drain the queue to avoid blocking consumers
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: Optional[float] = None) -> np.ndarray:
        """Read a chunk of audio PCM data (int16, mono) from the queue.

        Args:
            timeout: Seconds to wait for a chunk. If None, block indefinitely.

        Returns:
            A numpy array of shape (samples,) with dtype int16.

        Raises:
            queue.Empty: If the timeout elapses before a chunk is available.
        """
        return self._queue.get(timeout=timeout)

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def is_running(self) -> bool:
        """Return True if the stream is active."""
        return self._running.is_set() and self._stream is not None and self._stream.active