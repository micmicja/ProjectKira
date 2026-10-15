"""Exercise the full play() path against a real device and measure real-time pacing."""
import sys, time, asyncio
sys.path.insert(0, 'C:/code/ProjectKira')
import core.config as C
from core.audio.tts import TTS, chunk_text

s = C.load_settings(C.DEFAULT_CONFIG_FILE)
tts = TTS(s.tts, device=None, loop=asyncio.new_event_loop())
print('engine:', tts.engine_name)

TEXT = ("Okay. Here is a reply with several sentences in it. I am checking that each one "
        "is synthesised and played before the next one starts. This should take several "
        "seconds of real speaking time.")
chunks = chunk_text(TEXT)
print(f'{len(TEXT)} chars -> {len(chunks)} chunks:')
for i, c in enumerate(chunks):
    print(f'   [{i}] {len(c):3d} chars  {c[:44]!r}')

t0 = time.perf_counter()
ok = asyncio.run(tts.play(TEXT))
elapsed = time.perf_counter() - t0
print(f'play() -> {ok} in {elapsed:.2f}s for {len(chunks)} chunks')

# 24 words at a normal speaking pace is roughly 10-16 seconds of audio.
expected_lo, expected_hi = 7.0, 22.0
if elapsed < expected_lo:
    print(f'SUSPICIOUSLY FAST ({elapsed:.1f}s) -- audio may be missing or truncated')
if elapsed > expected_hi:
    print(f'VERY SLOW ({elapsed:.1f}s) -- latency budget is blown')
if expected_lo <= elapsed <= expected_hi:
    print(f'DURATION PLAUSIBLE ({elapsed:.1f}s)')
assert ok is True, 'play reported failure'

# A repeat must also work, proving the stream is reusable.
t0 = time.perf_counter()
ok2 = asyncio.run(tts.play('Repeated. Fine.'))
print(f'second play() -> {ok2} in {time.perf_counter()-t0:.2f}s')
assert ok2 is True

tts.close()
print('PLAY OK')
