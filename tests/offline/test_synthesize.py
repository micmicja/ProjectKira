"""Synthesis correctness: duration, container format, degenerate inputs."""
import sys, asyncio, io, time
sys.path.insert(0, 'C:/code/ProjectKira')
import numpy as np
import soundfile as sf
import core.config as C
from core.audio.tts import TTS, chunk_text


def _setup():
    s = C.load_settings(C.DEFAULT_CONFIG_FILE)
    return TTS(s.tts, device=None, loop=asyncio.new_event_loop())


def test_synthesize_duration_and_format():
    tts = _setup()
    text = ("Okay. Goodbye. And a final sentence to test the tail fragment, which is often "
            "the shortest one and is where a naive decoder tends to break on a stream.")
    assert len(chunk_text(text)) >= 2

    for chunked in (False, True):
        t0 = time.perf_counter()
        wav = asyncio.run(tts.synthesize(text, chunked=chunked))
        elapsed = time.perf_counter() - t0
        assert len(wav) > 100_000, f'{len(wav)} bytes is not a real utterance'

        with sf.SoundFile(io.BytesIO(wav)) as f:
            rate, channels, frames = f.samplerate, f.channels, f.frames
        duration = frames / rate

        # Two sentences of ~30 words total. Real speech runs ~150-180 wpm, so this
        # should land in the 8-20s range. Outside it means the audio is inflated
        # (per-fragment decode bug) or truncated (tail fragment dropped).
        assert 8.0 < duration < 20.0, f'chunked={chunked}: {duration:.2f}s is implausible'
        assert rate == 24000, rate
        assert channels == 1
        print(f'chunked={chunked}: {duration:.2f}s audio in {elapsed:.2f}s wall')

    tts.close()


def test_chunked_and_unchunked_agree_in_length():
    """Splitting into sentences must not change how long the reply takes to speak."""
    tts = _setup()
    text = ("One sentence here. Then a second one, and a third to finish off. "
            "Enough material that splitting makes a visible difference if anything breaks.")
    a = asyncio.run(tts.synthesize(text, chunked=False))
    b = asyncio.run(tts.synthesize(text, chunked=True))
    dur_a = sf.info(io.BytesIO(a)).frames / sf.info(io.BytesIO(a)).samplerate
    dur_b = sf.info(io.BytesIO(b)).frames / sf.info(io.BytesIO(b)).samplerate
    # Independent requests to a network TTS service are not byte-identical, but the
    # spoken length should agree to well under 20%.
    assert abs(dur_a - dur_b) / dur_a < 0.20, f'{dur_a:.2f}s vs {dur_b:.2f}s'
    print(f'unchunked {dur_a:.2f}s vs chunked {dur_b:.2f}s')
    tts.close()


def test_degenerate_inputs_never_raise():
    tts = _setup()
    for bad in ['', '   ', '!!!', '   .   ', '\n\n\t\n', '...']:
        out = asyncio.run(tts.synthesize(bad))
        assert out == b'', f'{bad!r} produced {len(out)} bytes'
    tts.close()


def test_empty_text_is_free():
    tts = _setup()
    assert asyncio.run(tts.play('')) is True
    assert asyncio.run(tts.synthesize('')) == b''
    tts.close()


if __name__ == '__main__':
    test_synthesize_duration_and_format()
    test_chunked_and_unchunked_agree_in_length()
    test_degenerate_inputs_never_raise()
    test_empty_text_is_free()
    print('SYNTHESIZE OK')
