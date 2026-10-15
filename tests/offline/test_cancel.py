"""Barge-in test: cancel() mid-utterance must stop audio promptly and be reusable."""
import sys, time, asyncio
sys.path.insert(0, 'C:/code/ProjectKira')
import core.config as C
from core.audio.tts import TTS

s = C.load_settings(C.DEFAULT_CONFIG_FILE)
tts = TTS(s.tts, device=None, loop=asyncio.new_event_loop())

LONG = ("This is a long reply that should take a while to speak out loud. It has enough "
        "sentences that we can interrupt it partway through and verify the interruption "
        "took effect quickly rather than letting the remainder finish playing. Continue "
        "until the very end so there is plenty of audio in flight when we cut it.")

async def run():
    loop = asyncio.get_running_loop()

    async def play_in_background():
        return await tts.play(LONG)

    task = asyncio.create_task(play_in_background())
    # Give it time to synthesise the first chunk and get audio going.
    await asyncio.sleep(4.0)

    t0 = time.perf_counter()
    tts.cancel()
    try:
        result = await asyncio.wait_for(task, timeout=10)
    except asyncio.TimeoutError:
        result = 'TIMEOUT'
    cancel_ms = (time.perf_counter() - t0) * 1000

    print(f'cancelled after 4s of a ~25s utterance')
    print(f'play() resolved to {result!r} {cancel_ms:.0f} ms after cancel()')
    return result

result = asyncio.run(run())
assert result is False, f'cancelled play() should return False, got {result!r}'
print('cancel returned False as expected')

# The stream must still be usable after an interruption.
t0 = time.perf_counter()
ok = asyncio.run(tts.play('And I am speaking again. This proves the stream survived.'))
print(f'post-cancel play() -> {ok} in {time.perf_counter()-t0:.2f}s')
assert ok is True

tts.close()
print('BARGE-IN OK')
