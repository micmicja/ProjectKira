"""Verify PlaybackStream pacing, drain detection, and cancellation on a real device.

Contract under test:
  * write() is non-blocking up to the ring-buffer capacity, then blocks for the overflow.
    That makes it a natural pacing primitive for sentence-sized writes.
  * flush() blocks until the last enqueued sample has been played.
  * cancel() aborts a blocked write from another thread and stops playback promptly.
"""
import sys, time, asyncio, threading
sys.path.insert(0, 'C:/code/ProjectKira')
import numpy as np
from core.audio.tts import PlaybackStream

RATE = 24000


def silent(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


ps = PlaybackStream(device=None, sample_rate=RATE, channels=1)
print('ready:', ps.ready, '| buffer latency:', ps._latency_s, 's')
assert ps.ready

# 1. A write smaller than the ring buffer must return immediately.
buf_s = ps._latency_s
t0 = time.perf_counter()
assert ps.write(silent(buf_s * 0.25)) is True
small_ms = (time.perf_counter() - t0) * 1000
print(f'write({buf_s*0.25:.2f}s) -> {small_ms:.0f} ms (buffer holds {buf_s:.2f}s)')
assert small_ms < 500

# 2. A large write blocks roughly by the overflow above the buffer capacity.
ps.flush(timeout_s=5)  # drain first so the baseline is clean
t0 = time.perf_counter()
assert ps.write(silent(2.0)) is True
big_ms = (time.perf_counter() - t0) * 1000
print(f'write(2.0s) blocked {big_ms:.0f} ms')
assert big_ms > (2.0 - buf_s) * 1000 * 0.5, 'expected significant blocking on oversized write'

# 3. flush() must wait for the queued audio to finish playing.
t0 = time.perf_counter()
assert ps.flush(timeout_s=10) is True
print(f'flush() drained in {time.perf_counter()-t0:.2f}s')

# 4. flush() with nothing outstanding returns immediately.
t0 = time.perf_counter()
assert ps.flush(timeout_s=5) is True
print(f'flush(empty) in {(time.perf_counter()-t0)*1000:.0f} ms')

# 5. cancel() from another thread must unstick a blocked write() call.
writer_error = {}

def _write():
    try:
        result = ps.write(silent(30.0))  # far more than the buffer holds
        writer_error['result'] = result
    except BaseException as exc:  # a torn-down stream may raise; that is acceptable
        writer_error['exc'] = f'{type(exc).__name__}: {exc}'

t0 = time.perf_counter()
th = threading.Thread(target=_write, daemon=True)
th.start()
time.sleep(0.5)  # let the write get stuck on the full buffer
assert not ps.cancelled, 'cancel flag should be clear before cancel()'
ps.cancel()
th.join(timeout=5)
elapsed = time.perf_counter() - t0
print(f'cancel() unblocked writer after {elapsed:.2f}s: {writer_error}')
assert th.is_alive() is False, 'write() was not released by cancel()'
assert elapsed < 5.0
assert ps.cancelled

# 6. After cancel, write/flush report failure rather than crashing.
assert ps.write(silent(1.0)) is False
assert ps.flush(timeout_s=2) is False
print('post-cancel calls handled cleanly')

# 7. reset() restores the stream for the next utterance.
ps.reset()
assert ps.cancelled is False
assert ps.write(silent(0.3)) is True
assert ps.flush(timeout_s=5) is True
print('post-reset playback works')

ps.close()
print('PLAYBACK STREAM OK')
