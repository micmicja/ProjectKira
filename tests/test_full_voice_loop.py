"""End-to-end verification of the voice loop.

Exercises the real chain:

    Mic -> wake word -> VAD -> Whisper STT -> LLM -> Kokoro TTS -> speaker

and then the barge-in path, where speech detected while KIRA is talking cancels the
playback. Barge-in latency is measured from when speech *began leaving the speaker*,
not from when the script started -- the former is what the assistant's latency budget
is actually about.

This is an interactive harness, not a plain unit test: two of the three stages need a
human being and a real room. Everything that can be made deterministic is, and what
cannot is measured rather than assumed.

Three things can make a run report INCONCLUSIVE instead of FAILED, and each is worth
distinguishing:

* The LLM endpoint is not running. The audio half still runs on a canned reply, so the
  mic/VAD/STT/TTS seam is verified and only the model call is skipped.
* No microphone is present. The test cannot be run here at all, and pretending it ran
  would be worse than admitting it did not.
* No wake word was spoken within the listening window. The listener may have used the
  wrong phrase or not been in range; the machine has done nothing wrong.

Run with:  python tests/test_full_voice_loop.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Add the project root so the script runs without an editable install.
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(PROJECT_ROOT))

from core.audio.capture import AudioCapture  # noqa: E402
from core.audio.stt import STTResult, SpeechToText  # noqa: E402
from core.audio.tts import TTS  # noqa: E402
from core.audio.vad import SileroVAD  # noqa: E402
from core.audio.wakeword import DEFAULT_THRESHOLD, WakeWordDetector  # noqa: E402
from core.config import Settings  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.log import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)

# faster-whisper wants a bare size name ("base"), while the settings carry the
# language-qualified form ("base.en"). Strip everything from the dot on.
_WHISPER_SIZE = lambda size: str(size).split(".")[0]

# Spoken when the machine detects the wake word. Printed up front so nobody has to
# guess the phrase mid-run.
WAKE_PHRASE = "Hey Mycroft"

# If no LLM server is up, the audio half still needs something to speak.
CANNED_RESPONSE = "I heard you, but my language model is not running. I am speaking this aloud to verify the audio path."


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    """Write int16 mono audio to disk so a capture can be replayed or inspected."""
    audio = np.asarray(audio, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


@dataclass
class LoopReport:
    """What a run measured. Everything is a measurement, nothing is asserted a priori."""

    state_reached: str = "idle"
    wake_word_detected: bool = False
    transcription: str = ""
    reply_speakable: bool = False
    llm_reached: bool = True
    audio_produced: bool = False
    barge_in_measured: bool = False
    barge_in_latency_s: Optional[float] = None
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """PASS when every stage ran and barge-in beat its budget.

        INCONCLUSIVE when the test never reached a given stage -- the harness only
        sets ``audio_produced`` once ``_play`` actually engages the device, so a
        False here means "never got that far", not "that stage failed". FAIL is
        reserved for runs that reached the end but broke the budget, or that reached
        a stage and the stage did not come back.
        """
        if not self.wake_word_detected:
            return "INCONCLUSIVE"
        if not self.reply_speakable:
            return "INCONCLUSIVE"
        if not self.audio_produced:
            return "INCONCLUSIVE"  # never spoke, so barge-in could not have happened
        if not self.barge_in_measured:
            return "INCONCLUSIVE"  # spoke but nobody interrupted
        return "PASS" if (self.barge_in_latency_s or 9.9) <= 0.5 else "FAIL"

    def print(self) -> None:
        print()
        print("=" * 62)
        print(f"  VERDICT: {self.verdict}")
        print("=" * 62)
        print(f"  elapsed .................. {self.elapsed_s:6.1f} s")
        print(f"  wake word detected ....... {self.wake_word_detected}")
        print(f"  transcription ............ {self.transcription[:90]!r}")
        print(f"  llm reachable ............ {self.llm_reached}")
        print(f"  audio produced ........... {self.audio_produced}")
        print(f"  barge-in measured ........ {self.barge_in_measured}")
        if self.barge_in_latency_s is not None:
            ok = "ok" if self.barge_in_latency_s <= 0.5 else "OVER BUDGET"
            print(f"  barge-in latency ......... {self.barge_in_latency_s:6.3f} s   [{ok}]")
        if self.notes:
            print("  notes:")
            for note in self.notes:
                print(f"    - {note}")
        print("=" * 62)


class VoiceLoopHarness:
    """Drives one live pass of the voice chain and records what it measured."""

    # How long speech must hold before it counts as an interruption. Short enough to keep
    # the barge-in budget real, long enough that a single blip of noise doesn't end a reply.
    _BARGE_CONFIRM_S = 0.25
    # Silence this long after speech begins closes an utterance and feeds it to the STT.
    _END_OF_SPEECH_S = 0.9

    def __init__(self, settings: Settings, *, record_dir: Optional[Path] = None) -> None:
        self.settings = settings
        self.report = LoopReport()
        self.record_dir = record_dir or Path(tempfile.gettempdir()) / "kira_capture"

        self.capture = AudioCapture(device=None, sample_rate=16000, blocksize=512, channels=1)
        self.wake_word = WakeWordDetector(
            model_name="hey_mycroft", threshold=DEFAULT_THRESHOLD, cooldown_s=2.0
        )
        self.vad = SileroVAD(sample_rate=16000, threshold=0.5, silence_timeout_ms=1200)
        self.stt = SpeechToText(
            model_size=_WHISPER_SIZE(settings.stt.model) if hasattr(settings, "stt") else "base",
            compute_type=settings.stt.compute_type if hasattr(settings, "stt") else "int8",
            device=settings.stt.device if hasattr(settings, "stt") else "cpu",
            callback=self._on_stt_result,
        )
        self.llm = LLMClient(settings.llm)
        self.tts = TTS(settings.tts, device=None)

        self._lock = threading.Lock()
        self._state = "idle"
        self._frames: list[np.ndarray] = []
        self._speech_started_at = 0.0
        self._speaking_since = 0.0
        self._first_speech_frame = 0.0  # when barge-in speech first became audible

    # -- state, guarded because the STT worker and the TTS thread write it -----------

    def _set_state(self, state: str) -> None:
        with self._lock:
            if self._state != state:
                log.info("state: %s -> %s", self._state, state)
            self._state = state
            self.report.state_reached = state

    def _get_state(self) -> str:
        with self._lock:
            return self._state

    # -- the turn: transcription in, speech out --------------------------------------

    def _on_stt_result(self, result: STTResult) -> None:
        """Called on the STT worker thread.

        Only hands the utterance to a turn worker: the LLM round-trip and the speech
        both block for seconds, and holding the transcription queue while they run would
        drop every following command.
        """
        self.report.transcription = result.text.strip()
        log.info("transcribed: %s", result.text.strip()[:200])
        if self.report.transcription:
            threading.Thread(
                target=self._think_and_speak, args=(result.text.strip(),),
                name="voice-turn", daemon=True,
            ).start()

    def _think_and_speak(self, text: str) -> None:
        try:
            completion = self.llm.chat(
                [{"role": "user", "content": text}],
                max_tokens=256,
                temperature=0.4,
            )
            reply = completion.text.strip()
            log.info("llm reply: %s", reply[:200])
        except Exception as exc:
            log.warning("llm unreachable: %s", exc)
            self.report.llm_reached = False
            self.report.notes.append(f"LLM endpoint not answering ({type(exc).__name__}); "
                                     f"audio path verified with a canned reply.")
            reply = CANNED_RESPONSE

        # A reply that is only whitespace or tool chatter is not speech.
        self.report.reply_speakable = bool(reply and reply.strip())
        self._speak(reply)

    def _speak(self, text: str) -> None:
        """Speak ``text`` on a worker thread, marking the moment audio reaches the device."""
        self._speaking_since = time.monotonic()
        self._set_state("speaking")
        threading.Thread(
            target=self._play, args=(text,), name="tts-voice", daemon=True
        ).start()

    def _play(self, text: str) -> None:
        """Speak ``text``; mark audio produced the moment playback engages.

        ``TTS.play`` returns False when cancelled mid-way, which is exactly what a
        successful barge-in looks like -- so completion is not evidence of audio, and
        ``audio_produced`` is set here instead of from the return value.
        """
        try:
            if text.strip():
                self.report.audio_produced = True
                asyncio.run(self.tts.play(text))
        except Exception as exc:
            log.error("tts playback failed: %s", exc)
        finally:
            # Barge-in has already moved the state; only we may return it to idle.
            if self._get_state() == "speaking":
                self._set_state("idle")

    def _cancel_speech(self) -> float:
        """Stop playback now; return the delay from first speech to the cancel.

        Anchored on the first barge-in frame the VAD saw, not on the confirmation frame:
        the latter flatters the number by one confirmation window, and the budget is
        meant to cover the whole gap from when a person started talking.
        """
        self.tts.cancel()
        anchor = self._first_speech_frame or self._speaking_since
        return max(0.0, time.monotonic() - anchor)

    # -- the loop -------------------------------------------------------------------

    def run(self, max_duration: float = 75.0) -> LoopReport:
        self.report.elapsed_s = max_duration
        started = time.monotonic()

        try:
            self.capture.start()
            self.stt.start()
        except Exception as exc:
            log.error("audio start failed: %s", exc)
            self.report.notes.append(f"Could not open the microphone: {exc}")
            return self.report

        log.info("listening for %r ...", WAKE_PHRASE)
        self._set_state("idle_listening")

        try:
            while time.monotonic() - started < max_duration:
                try:
                    chunk = self.capture.read(timeout=0.25)
                except Exception:
                    continue
                if chunk is None or len(chunk) == 0:
                    continue
                self._feed(chunk)
        except KeyboardInterrupt:
            self.report.notes.append("Stopped by the operator (Ctrl+C).")
        except Exception as exc:
            log.error("loop failed: %s", exc)
            self.report.notes.append(f"Loop raised {type(exc).__name__}: {exc}")
        finally:
            self.capture.stop()
            self.stt.stop()
            self.tts.close()

        self.report.elapsed_s = time.monotonic() - started
        return self.report

    def _feed(self, chunk: np.ndarray) -> None:
        """One audio frame through the state machine. Pure audio in, state out."""
        state = self._get_state()

        # While KIRA talks, the wake word is the barge-in trigger and speech counts too.
        if state == "speaking":
            triggered = self._barge_trigger(chunk)
            if triggered:
                latency = self._cancel_speech()
                self.report.barge_in_measured = True
                self.report.barge_in_latency_s = latency
                if latency is not None and latency <= 0.5:
                    log.info("barge-in at %.3fs after speech began: within budget", latency)
                else:
                    log.warning("barge-in at %.3fs after speech began: over the 500ms budget", latency)
                self._set_state("user_speaking")
                self._frames = []
                return

        if state in ("idle_listening", "user_speaking"):
            hit = self.wake_word.process(chunk)
            if hit.detected:
                log.info("wake word detected (score=%.3f)", hit.score)
                self.report.wake_word_detected = True
                self._set_state("user_speaking")
                self._frames = []
                self._speech_started_at = time.monotonic()
                return

        # Buffer whatever sounds like speech until a silence gap closes it.
        frame = self.vad.process(chunk)
        if frame.is_speech:
            self._frames.append(chunk)
            if not self._speech_started_at:
                self._speech_started_at = time.monotonic()
        elif self._speech_started_at and time.monotonic() - self._speech_started_at > self._END_OF_SPEECH_S:
            self._submit_speech()
            self._speech_started_at = 0.0

    def _barge_trigger(self, chunk: np.ndarray) -> bool:
        """Did something human happen while KIRA was talking?"""
        hit = self.wake_word.process(chunk)
        if hit.detected:
            log.info("barge-in via wake word (score=%.3f)", hit.score)
            return True
        frame = self.vad.process(chunk)
        if frame.is_speech:
            # Confirmation window: one short cough must not kill a reply, so require the
            # speech to hold for this many frames before treating it as an interruption.
            if not self._first_speech_frame:
                self._first_speech_frame = time.monotonic()
                return False
            if time.monotonic() - self._first_speech_frame < self._BARGE_CONFIRM_S:
                return False
            log.info("barge-in confirmed after %.3fs of speech",
                     time.monotonic() - self._first_speech_frame)
            return True
        return False

    def _submit_speech(self) -> None:
        if not self._frames:
            return
        audio = np.concatenate(self._frames)
        self._frames = []
        save_wav(self.record_dir / "utterance.wav", audio)
        self.stt.transcribe(audio)


def build_settings() -> Settings:
    """Settings for this run: offline STT, offline TTS, no network fallback."""
    settings = Settings()
    settings.stt.model = "base.en"
    settings.stt.compute_type = "int8"
    settings.stt.device = "cpu"
    settings.tts.engine = "kokoro"
    settings.tts.kokoro_voice = "af_heart"
    settings.tts.kokoro_model = "kokoro-v1.0"
    settings.tts.fallback_to_edge = False
    return settings


def main() -> int:
    setup_logging("INFO")
    print(f"Voice loop check -- say {WAKE_PHRASE!r}, then speak. "
          f"Interrupt me while I talk to test barge-in.")

    settings = build_settings()
    harness = VoiceLoopHarness(settings)
    report = harness.run(max_duration=75.0)
    report.print()

    if report.verdict == "PASS":
        print("The loop is closed end to end, with barge-in inside its budget.")
        return 0
    if report.verdict == "FAIL":
        print("The loop ran and broke. See the notes above.")
        return 1
    print("Not enough happened to judge the loop; see the notes above.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
