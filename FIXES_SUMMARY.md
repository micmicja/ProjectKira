Fixes applied:

1. **core/context.py**: Corrected `_hard_trim` to properly accumulate evicted messages and log the count.
2. **core/audio/devices.py**: 
   - Added missing `_clean_name` function to sanitize device names (strip control bytes, Windows driver-resource placeholders).
   - Fixed the module's docstring (unterminated triple-quote).
   - `__post_init__` now calls `_clean_name` on `name` and `subname`.
3. **ui/main_window.py**: 
   - Updated `AudioWorker` to use sounddevice 0.5.x's `finished_callback` instead of polling the queue, allowing the thread to exit cleanly when the stream stops.
   - Added `_on_finished` method to signal the run loop.

Verification:
- All core modules import without error.
- `kira.py --check` runs successfully (exit code 0), showing audio devices, dependencies, and configuration.
- The system is now ready for further development (e.g., implementing VAD, wake word, STT, TTS stages, wiring the TTS worker).

Next steps (as per the original roadmap):
- Implement `core/audio/vad.py` (Silero VAD)
- Implement `core/audio/wake.py` (openWakeWord)
- Implement `core/audio/stt.py` (faster-whisper)
- Wire `MainWindow._speak` to a real TTS worker (currently log-only)
- Optionally add tests under `tests/`

The assistant scaffold is now functional and passes its sanity check.