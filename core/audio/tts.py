from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import queue
import re
import time
from pathlib import Path
from typing import AsyncIterator, Iterator, Protocol

import numpy as np

from core.log import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Text handling
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"([.!?]+[\"')\]]*)\s+")
_MIN_CHARS = 24
_MAX_CHARS = 400


def _split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        pieces.append(text[start:match.start() + match.lastindex])
        start = match.end()
    pieces.append(text[start:])
    return [p.strip() for p in pieces if p.strip()]


def chunk_text(text: str, *, min_chars: int = _MIN_CHARS, max_chars: int = _MAX_CHARS) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = _split_sentences(text)
    if not parts:
        parts = [text]

    out: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = None
            for comma in (i for i in range(len(part) - 1, min(max_chars // 2 - 1, 0), -1)
                          if part[i] in ",;:"):
                cut = comma + 1
                break
            if cut is None:
                cut = max_chars
            out.append(part[:cut].strip())
            part = part[cut:].strip()
            if not part:
                break

        if not part:
            continue
        if out and len(out[-1]) + 1 + len(part) <= min_chars:
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)

    return out or [text]


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

class TTSBackend(Protocol):
    name: str
    voice: str

    def synthesize_sync(self, text: str) -> bytes:
        ...

    def stream_async(self, text: str) -> AsyncIterator[tuple[bytes, str]]:
        ...


def _has(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


class EdgeTTSEngine:
    def __init__(self, *, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Run:  pip install edge-tts"
            ) from exc
        self.name = "edge"
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    def _communicate(self, text: str):
        import edge_tts
        return edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)

    def synthesize_sync(self, text: str) -> bytes:
        import asyncio
        async def _run() -> bytes:
            buf = bytearray()
            async for chunk in self._communicate(text).stream():
                if chunk["type"] == "audio" and chunk.get("data"):
                    buf.extend(chunk["data"])
            return bytes(buf)
        return asyncio.run(_run())

    def stream_async(self, text: str) -> AsyncIterator[tuple[bytes, str]]:
        comm = self._communicate(text)
        async def _gen() -> AsyncIterator[tuple[bytes, str]]:
            try:
                async for chunk in comm.stream():
                    if chunk["type"] == "audio" and chunk.get("data"):
                        yield chunk["data"], "audio"
            finally:
                _close_communicate(comm)
                yield b"", "done"
        return _gen()


def _close_communicate(comm) -> None:
    close = getattr(comm, "close", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        close()


class KokoroEngine:
    def __init__(self, *, voice: str, model: str = "kokoro-v1.0") -> None:
        try:
            import kokoro_onnx
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx is not installed. Run:  pip install kokoro-onnx"
            ) from exc
        self.name = "kokoro"
        self.voice = voice
        self.model = model
        self._kokoro_onnx = kokoro_onnx
        model_path, voices_path = self._resolve_paths()
        self._session = kokoro_onnx.Kokoro(model_path, voices_path)

    @staticmethod
    def _resolve_paths() -> tuple[str, str]:
        import os
        env_model = os.environ.get("KIRA_KOKORO_MODEL")
        env_voices = os.environ.get("KIRA_KOKORO_VOICES")
        if env_model and os.path.exists(env_model) and env_voices and os.path.exists(env_voices):
            return env_model, env_voices

        roots = [
            Path.home() / ".cache" / "kira" / "kokoro",
            Path(os.getcwd()) / "data" / "models" / "kokoro",
            Path(__file__).resolve().parents[2] / "data" / "models" / "kokoro",
        ]

        found_model = found_voices = None
        for root in roots:
            if not root.is_dir():
                continue
            candidates_model = [p for p in root.glob("kokoro-v1*.onnx") if p.is_file()]
            if found_model is None and candidates_model:
                found_model = str(candidates_model[0])
            for pat in ("voices-v1*.bin", "voices*.npy", "voices*.bin"):
                for p in sorted(root.glob(pat)):
                    if p.is_file():
                        found_voices = str(p)
                        break
            if found_model and found_voices:
                break

        if not found_model:
            raise RuntimeError(
                "kokoro-v1.0.onnx not found. Download it to data/models/kokoro/ from "
                "https://github.com/thewh1teagle/kokoro-onnx/releases "
                "(model-files-v1.0) or set KIRA_KOKORO_MODEL."
            )
        if not found_voices:
            raise RuntimeError(
                f"voice table not found next to {found_model}. Download voices-v1.0.bin "
                "from the same release, or set KIRA_KOKORO_VOICES."
            )
        return found_model, found_voices

    def synthesize_sync(self, text: str) -> bytes:
        import io
        import soundfile as sf
        audio, rate = self._session.create(text, self.voice)
        buf = io.BytesIO()
        with sf.SoundFile(buf, "wb", samplerate=int(rate), channels=1,
                          format="WAV", subtype="PCM_16") as f:
            f.write(_pcm_to_s16(audio))
        return buf.getvalue()

    def stream_async(self, text: str) -> AsyncIterator[tuple[bytes, str]]:
        def _gen() -> Iterator[tuple[bytes, str]]:
            yield self.synthesize_sync(text), "audio"
            yield b"", "done"
        return _async_gen(_gen())


async def _async_gen(sync_iter: Iterator[tuple[bytes, str]]) -> AsyncIterator[tuple[bytes, str]]:
    loop = asyncio.get_running_loop()
    q: queue.Queue[tuple[bytes, str] | None] = queue.Queue(maxsize=2)

    def _push() -> None:
        try:
            for item in sync_iter:
                q.put(item)
        finally:
            q.put(None)

    producer = loop.run_in_executor(None, _push)
    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield item
    finally:
        producer.result()


def make_engine(kind: str, *, voice: str, model: str = "kokoro-v1.0",
                rate: str = "+0%", pitch: str = "+0Hz") -> TTSBackend:
    if kind.strip().lower() == "kokoro":
        if _has("kokoro_onnx"):
            return KokoroEngine(voice=voice, model=model)
        log.warning("kokoro-onnx not installed; falling back to edge-tts")
    return EdgeTTSEngine(voice=voice, rate=rate, pitch=pitch)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #

def decode_audio_bytes(data: bytes, *, src_rate: int = 24000, to_rate: int | None = None) -> np.ndarray:
    if importlib.util.find_spec("soundfile") is None:
        raise RuntimeError("soundfile is not installed. Run:  pip install soundfile")
    import io
    import soundfile as sf

    if not data:
        return np.zeros(0, dtype=np.float32)

    with sf.SoundFile(io.BytesIO(data), "rb", format=None) as f:
        frames = f.read(dtype="float32")
        native_rate = int(f.samplerate)

    frames = _as_mono(frames)
    rate = int(to_rate or src_rate)
    if native_rate != rate:
        frames = resample_mono(frames, native_rate, rate)
    return np.asarray(frames, dtype=np.float32)


def _as_mono(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 1:
        return np.asarray(frames, dtype=np.float32)
    return np.mean(frames, axis=1).astype(np.float32)


def resample_mono(frames: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate <= 0 or to_rate <= 0 or len(frames) == 0:
        return np.asarray(frames, dtype=np.float32)
    if from_rate == to_rate:
        return np.asarray(frames, dtype=np.float32)

    x = np.asarray(frames, dtype=np.float64)
    ratio = to_rate / from_rate
    n_out = int(round(len(x) * ratio))
    if n_out <= 1:
        return np.zeros(n_out, dtype=np.float32)

    src_pos = np.linspace(0.0, len(x) - 1, n_out)
    lo = np.clip(np.floor(src_pos).astype(np.int64), 0, len(x) - 1)
    hi = np.clip(lo + 1, 0, len(x) - 1)
    frac = (src_pos - lo).astype(np.float64)
    return (x[lo] * (1.0 - frac) + x[hi] * frac).astype(np.float32)


# --------------------------------------------------------------------------- #
# Playback
# --------------------------------------------------------------------------- #

class PlaybackStream:
    _DRAIN_POLL_S = 0.015
    _DRAIN_TOLERANCE_S = 0.030

    def __init__(self, *, device: int | None = None, sample_rate: int = 24000,
                 channels: int = 1, latency_s: float = 0.180) -> None:
        self._sd = None
        self._stream = None
        self.device = device
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._latency_s = latency_s
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        self._close_stream()

    def reset(self) -> None:
        self._cancelled = False

    @property
    def ready(self) -> bool:
        if self._stream is not None:
            return True
        if self._sd is None:
            if importlib.util.find_spec("sounddevice") is None:
                return False
            try:
                import sounddevice as sd
                self._sd = sd
            except Exception:
                return False
        return True

    def _open(self) -> bool:
        if self._stream is not None:
            return True
        if not self.ready:
            return False
        sd = self._sd
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                device=self.device,
                channels=self.channels,
                dtype="float32",
                latency=self._latency_s,
            )
            self._stream.start()
            return True
        except Exception as exc:
            log.error("failed to open audio output: %s", exc)
            self._stream = None
            return False

    def write(self, frames: np.ndarray) -> bool:
        if self._cancelled:
            return False
        if not self._open():
            return False

        data = np.asarray(frames, dtype=np.float32)
        if self.channels == 1 and data.ndim == 1:
            data = data.reshape(-1, 1)
        if self.channels > 1:
            data = np.repeat(data, self.channels, axis=1)

        try:
            self._stream.write(data)
        except Exception as exc:
            if self._cancelled:
                return False
            log.error("audio write failed: %s", exc)
            return False
        return True

    def flush(self, timeout_s: float = 20.0) -> bool:
        if self._cancelled:
            return False
        stream = self._stream
        if stream is None:
            return True

        deadline = time.monotonic() + timeout_s
        while True:
            if self._cancelled:
                return False
            try:
                position = float(stream.time)
            except Exception:
                return True

            lag = position - time.monotonic()
            if lag <= self._DRAIN_TOLERANCE_S:
                return True
            if time.monotonic() >= deadline:
                log.warning("audio flush timed out with %.2fs still buffered", lag)
                return False
            time.sleep(self._DRAIN_POLL_S)

    def stop(self) -> None:
        self._close_stream()

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    def close(self) -> None:
        self._close_stream()
        self._sd = None


def _pcm_to_s16(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.zeros(0, dtype=np.int16)
    return np.clip(np.asarray(frames, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #

class TTS:
    def __init__(self, settings, *, device: int | None = None, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.settings = settings
        self.device = device
        self._loop = loop
        self.stream = PlaybackStream(
            device=device,
            sample_rate=int(getattr(settings, "tts_sample_rate", 24000) or 24000),
        )
        try:
            self.engine: TTSBackend | None = make_engine(
                str(getattr(settings, "engine", "edge")),
                voice=str(getattr(settings, "kokoro_voice",
                                  getattr(settings, "voice", "en-US-AriaNeural"))),
                model=str(getattr(settings, "kokoro_model", "kokoro-v1.0")),
                rate=str(getattr(settings, "rate", "+0%")),
                pitch=str(getattr(settings, "pitch", "+0Hz")),
            )
        except Exception as exc:
            if not bool(getattr(settings, "fallback_to_edge", True)):
                raise
            log.warning("primary TTS engine unavailable (%s); using edge-tts", exc)
            self.engine = EdgeTTSEngine(
                voice=str(getattr(settings, "edge_voice", "en-US-AriaNeural")),
                rate=str(getattr(settings, "rate", "+0%")),
                pitch=str(getattr(settings, "pitch", "+0Hz")),
            )

    @property
    def engine_name(self) -> str | None:
        return getattr(self.engine, "name", None)

    def close(self) -> None:
        self.stream.close()

    async def synthesize(self, text: str, *, chunked: bool = True) -> bytes:
        if self.engine is None:
            return b""
        text = (text or "").strip()
        if not text:
            return b""

        import io
        if importlib.util.find_spec("soundfile") is None:
            raise RuntimeError("soundfile is not installed. Run:  pip install soundfile")
        import soundfile as sf

        chunks = chunk_text(text) if chunked else [text]
        speed = float(getattr(self.settings, "speed", 1.0) or 1.0)

        bufs: list[np.ndarray] = []
        for idx, piece in enumerate(chunks):
            t0 = time.perf_counter()
            try:
                raw = bytearray()
                async for chunk_bytes, kind in self.engine.stream_async(piece):
                    if kind != "audio" or not chunk_bytes:
                        continue
                    raw.extend(chunk_bytes)
                if raw:
                    arr = decode_audio_bytes(bytes(raw))
                    if arr.size:
                        bufs.append(arr)
            except Exception as exc:
                log.warning("tts chunk %d/%d failed: %s", idx, len(chunks), exc)
                continue
            log.debug("tts chunk %d/%d (%d chars) %s", idx, len(chunks), len(piece),
                      _latency_ms(t0))

        if not bufs:
            return b""

        frames = np.concatenate(bufs)
        if abs(speed - 1.0) > 1e-3:
            frames = resample_mono(frames, int(24000 * speed), 24000)

        buf = io.BytesIO()
        with sf.SoundFile(buf, "wb", samplerate=24000, channels=1,
                          format="WAV", subtype="PCM_16") as f:
            f.write(_pcm_to_s16(frames))
        return buf.getvalue()

    def save(self, text: str, path: str | Path, *, chunked: bool = True) -> Path:
        out = Path(path)
        data = _run_coroutine(self.synthesize(text, chunked=chunked))
        if not data:
            raise RuntimeError("synthesis produced no audio")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    async def play(self, text: str) -> bool:
        if self.engine is None:
            log.warning("tts play skipped: no engine available")
            return False
        text = (text or "").strip()
        if not text:
            return True

        self.stream.reset()
        if not self.stream._open():
            return False

        chunks = chunk_text(text)
        all_ok = True
        try:
            for idx, piece in enumerate(chunks):
                if self.stream.cancelled:
                    all_ok = False
                    break
                t0 = time.perf_counter()
                frames = await self._synth_async(piece)
                if frames is None:
                    if idx == 0:
                        return False
                    all_ok = False
                    break
                if not await asyncio.to_thread(self.stream.write, frames):
                    all_ok = False
                    break
                log.debug("tts played chunk %d/%d (%d chars) %s", idx, len(chunks),
                          len(piece), _latency_ms(t0))
        finally:
            if not self.stream.cancelled:
                await asyncio.to_thread(self.stream.flush, 20.0)
            else:
                self.stream.stop()

        return all_ok

    async def _synth_async(self, text: str) -> np.ndarray | None:
        if self.engine is None:
            return None
        speed = float(getattr(self.settings, "speed", 1.0) or 1.0)
        try:
            raw = bytearray()
            async for chunk_bytes, kind in self.engine.stream_async(text):
                if kind != "audio" or not chunk_bytes:
                    continue
                raw.extend(chunk_bytes)
            if not raw:
                return None
            frames = decode_audio_bytes(bytes(raw), to_rate=self.stream.sample_rate)
            if frames.size == 0:
                return None
            if abs(speed - 1.0) > 1e-3:
                frames = resample_mono(frames, self.stream.sample_rate,
                                       int(self.stream.sample_rate * speed))
            return frames
        except Exception as exc:
            log.warning("tts synthesis failed for %r: %s", text[:40], exc)
            return None

    def cancel(self) -> None:
        self.stream.cancel()


def _latency_ms(start: float) -> str:
    return f"{(time.perf_counter() - start) * 1000:.0f} ms"


def _run_coroutine(coro) -> bytes:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


__all__ = [
    "EdgeTTSEngine",
    "KokoroEngine",
    "PlaybackStream",
    "TTS",
    "chunk_text",
    "decode_audio_bytes",
    "make_engine",
    "resample_mono",
]
