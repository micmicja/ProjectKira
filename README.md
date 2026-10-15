# KIRA - Local Voice Assistant

KIRA is a privacy-first, low-latency voice assistant for Windows that runs entirely locally. It features:

- OpenWakeWord wake word detection
- Silero VAD for speech activity detection
- Faster-Whisper for speech-to-text transcription
- Kokoro-TTS with Edge-TTS fallback for text-to-speech
- Local OpenAI-compatible LLM inference (qwen2.5-7b-instruct)

## Installation

### Prerequisites
- Windows 10/11
- Python 3.14.7
- Visual C++ Build Tools (for sounddevice dependencies)

### Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the assistant
```powershell
.\.venv\Scripts\python.exe kira.py
```

### Check system configuration
```powershell
.\.venv\Scripts\python.exe kira.py --check
```

## Features

- **Privacy-first**: All processing happens locally, no data sent to external servers
- **Low-latency**: Optimized for real-time voice interaction
- **Configurable**: Customizable via `data/kira.yaml`
- **Extensible**: Modular architecture for adding new features

## Project Structure

- `core/`: Core Python modules
  - `config.py`: Configuration management
  - `log.py`: Structured logging system
  - `llm.py`: LLM client and chat handling
  - `tools/`: Tool registry and verification utilities
  - `context.py`: Conversation context and summarization
- `core/audio/`: Audio processing modules
  - `devices.py`: Audio device enumeration and routing
  - `vad.py`: Speech activity detection
  - `wake.py`: Wake word detection
  - `stt.py`: Speech-to-text transcription
  - `tts.py`: Text-to-speech synthesis
- `ui/`: GUI components using PyQt6
- `tests/`: Test suite

## License

MIT License - see LICENSE file for details.