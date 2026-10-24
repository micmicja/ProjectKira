#!/usr/bin/env python
"""Pre-flight diagnostic for KIRA.

Checks the NVIDIA LLM endpoint and the Edge-TTS + Pygame audio pipeline
without launching the PyQt6 UI. Run with:

    .venv\\Scripts\\python diagnostic.py

Prints a per-check [ OK ] / [ FAILED ] line and exits non-zero if any
mandatory check fails."""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

# Exit codes: 0 all ok, 1 = LLM failed, 2 = audio failed (so scripts can act on which).
OK = 0
LLM_FAILED = 1
AUDIO_FAILED = 2


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    """Load tts/llm relevant keys from kira.yaml (or the template default)."""
    path = None
    for candidate in (
        PROJECT_ROOT / "data" / "kira.yaml",
        PROJECT_ROOT / "kira_config.yaml",
        Path.cwd() / "kira_config.yaml",
    ):
        if candidate.exists():
            path = candidate
            break
    if path is None:
        print("     [ WARN ] no config found; using built-in defaults")
        return {"llm": {}, "tts": {}}

    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"     [ WARN ] could not parse config {path}: {exc}")
        return {"llm": {}, "tts": {}}
    return data


def test_llm(cfg: dict) -> int:
    """Send a minimal prompt to the OpenAI-compatible endpoint."""
    llm = cfg.get("llm") or {}
    base_url = str(llm.get("server_url") or "http://127.0.0.1:11434/v1").rstrip("/")
    model = str(llm.get("model") or "")
    api_key = str(llm.get("api_key") or "")
    if not model:
        print("  [ FAILED ] LLM: no model configured in kira.yaml")
        return LLM_FAILED
    if not api_key:
        print("  [ FAILED ] LLM: no api_key configured in kira.yaml")
        return LLM_FAILED

    try:
        from openai import OpenAI
    except ImportError:
        print("  [ FAILED ] LLM: openai package not installed (pip install openai)")
        return LLM_FAILED

    print(f"  [ .. ] testing LLM {base_url}  model={model}")
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: API OK"}],
            max_tokens=8,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  [ FAILED ] LLM: {exc}")
        return LLM_FAILED

    print(f"  [ OK ] LLM: {text[:80]}")
    return OK


def test_audio(cfg: dict) -> int:
    """Synthesise a short phrase with Edge-TTS and play it via pygame.mixer."""
    print("  [ .. ] testing Edge-TTS -> pygame.mixer audio pipeline")
    try:
        import edge_tts
    except ImportError:
        print("  [ FAILED ] audio: edge-tts not installed (pip install edge-tts)")
        return AUDIO_FAILED
    try:
        import pygame
    except ImportError:
        print("  [ FAILED ] audio: pygame not installed (pip install pygame)")
        return AUDIO_FAILED

    try:
        # Synthesise a short phrase; 24 kHz is Edge-TTS native rate.
        raw: bytearray = bytearray()

        async def _fetch() -> None:
            comm = edge_tts.Communicate("Audio systems online.", "en-US-ChristopherNeural")
            try:
                async for chunk in comm.stream():
                    if chunk["type"] == "audio" and chunk.get("data"):
                        raw.extend(chunk["data"])
            finally:
                close = getattr(comm, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass

        asyncio.run(_fetch())
        if not raw:
            print("  [ FAILED ] audio: Edge-TTS produced no audio bytes")
            return AUDIO_FAILED

        # Play natively via pygame, from an in-memory buffer (no external player).
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=4096)
        pygame.mixer.music.load(io.BytesIO(bytes(raw)))
        pygame.mixer.music.play()
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)
        pygame.mixer.quit()
    except Exception as exc:
        print(f"  [ FAILED ] audio: {exc}")
        return AUDIO_FAILED

    print("  [ OK ] audio: Edge-TTS -> pygame playback succeeded")
    return OK


def main() -> int:
    print("KIRA pre-flight diagnostic")
    print("=" * 60)

    cfg = load_config()

    print("\n[1] LLM connection")
    llm_status = test_llm(cfg)

    print("\n[2] Audio pipeline (Edge-TTS -> pygame)")
    audio_status = test_audio(cfg)

    status = OK
    if llm_status == LLM_FAILED:
        status = LLM_FAILED
    if audio_status == AUDIO_FAILED and status == OK:
        status = AUDIO_FAILED

    print("\n" + "=" * 60)
    if status == OK:
        print("All checks passed.")
    else:
        print("One or more checks failed - see above.")
    return status


if __name__ == "__main__":
    sys.exit(main())