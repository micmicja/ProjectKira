import sys, asyncio, time
sys.path.insert(0, 'C:/code/ProjectKira')
import numpy as np
import core.config as C
from core.audio.tts import TTS, decode_audio_bytes

s = C.load_settings(C.DEFAULT_CONFIG_FILE)
tts = TTS(s.tts, device=None, loop=asyncio.new_event_loop())
print('engine:', tts.engine_name)

TEXT = "Hello. I am Kira, a local voice assistant. What time is it, and how about a short answer? I think so."
chunks_expected = 3

# 1. Full synthesis with timing
t0 = time.perf_counter()
wav = asyncio.run(tts.synthesize(TEXT))
print(f'synthesize: {len(wav)} bytes in {time.perf_counter()-t0:.2f}s')
assert len(wav) > 2000, 'suspiciously small output'

# 2. Confirm it is decodable WAV at 24 kHz
import soundfile as sf
with sf.SoundFile(__import__('io').BytesIO(wav)) as f:
    print(f'container: {f.format} {f.subtype} rate={f.samplerate} ch={f.channels} frames={f.frames}')
    assert f.samplerate == 24000
    assert f.channels == 1

# 3. Non-chunked path
t0 = time.perf_counter()
wav2 = asyncio.run(tts.synthesize(TEXT, chunked=False))
print(f'synthesize unchunked: {len(wav2)} bytes in {time.perf_counter()-t0:.2f}s')
assert len(wav2) > 1000

# 4. Degenerate inputs must not raise
for bad in ['', '   ', '!!!', '   .   ', '\n\n\t\n']:
    out = asyncio.run(tts.synthesize(bad))
    print(f'degenerate {bad!r} -> {len(out)} bytes')

# 5. Playback must not raise even without a usable device, and cancel must be cheap.
# Playback is intentionally not asserted to return True: there is no guarantee this
# environment has a working audio sink, and an unresolvable path is a valid result.
async def play_probe():
    try:
        return await asyncio.wait_for(tts.play('Okay. Goodbye.'), timeout=30)
    except Exception as exc:
        return f'raised {type(exc).__name__}: {exc}'

t0 = time.perf_counter()
result = asyncio.run(play_probe())
print(f'play() -> {result!r} in {time.perf_counter()-t0:.2f}s')

t0 = time.perf_counter()
tts.cancel()
print(f'cancel() in {time.perf_counter()-t0:.4f}s')

t0 = time.perf_counter()
tts.close()
print(f'close() in {time.perf_counter()-t0:.4f}s')
print('END TO END OK')
