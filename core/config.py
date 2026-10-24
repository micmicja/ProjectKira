"""Configuration for KIRA.

YAML-backed settings over dataclass defaults, so KIRA runs with zero config and every
value is still overridable. Precedence, lowest to highest:

    code defaults  <  kira.yaml  <  KIRA_* env vars  <  CLI flags

Config discovery (first hit wins):

    <project>/data/kira.yaml        project-local; the default while developing
    <cwd>/kira_config.yaml          legacy / hand-placed next to kira.py
    <user config dir>/config.yaml   user-level fallback
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

APP_NAME = "kira"

# Windows user state lives under %APPDATA%; other platforms use XDG.
if os.name == "nt":
    USER_CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
else:
    USER_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_FILE = PROJECT_DATA_DIR / "kira.yaml"
DEFAULT_LOG_DIR = PROJECT_DATA_DIR / "logs"

# Sections in file order, and the path-typed fields within each.
_SECTION_ORDER = ("audio", "wake", "stt", "tts", "llm", "agent", "logging")
_PATH_FIELDS = {
    "wake": ("model_path",),
    "stt": ("download_root",),
    "tts": ("cache_dir",),
    "logging": ("dir",),
}


# --------------------------------------------------------------------------- #
# Section definitions
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class AudioSettings:
    """Capture/playback routing and the signal parameters of the pipeline."""

    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = 16_000  # 16 kHz mono: what the wake word and VAD models expect.
    channels: int = 1
    block_ms: int = 40  # 640 frames — the openWakeWord inference window.
    # Audio retained before the wake word fires, so "Kira, what time is it" keeps the
    # whole command rather than losing its first words.
    wake_prebuffer_s: float = 1.5
    vad_threshold: float = 0.5  # Silero speech probability threshold.
    vad_min_silence_s: float = 0.7  # Silence this long closes an utterance.
    vad_min_speech_s: float = 0.25  # Blips shorter than this are ignored.
    listen_timeout_s: float = 8.0  # Give up this long after "Kira" with no speech.
    echo_cancel: bool = True
    tts_sample_rate: int = 24_000  # Kokoro's native rate; resample on playback.


@dataclass(slots=True)
class WakeSettings:
    """openWakeWord. Models are lazy-loaded on first use."""

    enabled: bool = True
    model_path: Path | None = None  # *.onnx; None => bundled pretrained model
    sensitivity: float = 0.5  # 0..1; higher = more triggers, more false positives
    detection_threshold: float = 0.5
    min_score: float = 0.25
    language_model: str = "en_US"
    # Re-arm only after this much silence so one utterance cannot fire twice.
    rearm_silence_s: float = 1.2


@dataclass(slots=True)
class STTSettings:
    """faster-whisper (CTranslate2) — CPU by default, GPU when available."""

    enabled: bool = True
    model: str = "base.en"  # tiny.en | base.en | small.en | medium.en | large-v3
    device: str = "auto"  # auto | cpu | cuda | rocm | metal
    compute_type: str = "auto"  # auto | int8 | int8_float16 | float16 | float32
    beam_size: int = 1  # 1 is fastest; raise to 5 for accuracy on hard audio.
    vad_filter: bool = True
    language: str | None = "en"
    download_root: Path | None = None  # None => ~/.cache/huggingface


@dataclass(slots=True)
class TTSSettings:
    """Edge-TTS primary (network-based, no local CPU load)."""

    engine: str = "edge"  # kokoro | edge
    kokoro_voice: str = "af_heart"
    kokoro_model: str = "kokoro-v1.0"
    speed: float = 1.0
    edge_voice: str = "en-US-ChristopherNeural"  # Professional male J.A.R.V.I.S. tone
    rate: str = "+0%"
    pitch: str = "+0Hz"
    fallback_to_edge: bool = True
    cache_dir: Path | None = None
    stream: bool = True  # Feed sentence chunks to the speaker as they arrive.


@dataclass(slots=True)
class LLMSettings:
    """OpenAI-compatible endpoint: LM Studio, llama.cpp server, vLLM, Ollama."""

    server_url: str = "http://127.0.0.1:11434/v1"
    model: str = "llama3.2:3b"
    # Deliberately low: local instruct models beat their defaults at very low temp.
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = True
    timeout_s: float = 60.0
    api_key: str = "ollama"  # Local servers ignore it; the SDK requires one.
    # Extra body fields sent with every request — for server knobs the OpenAI SDK
    # does not expose, e.g. {"chat_template_kwargs": {"reasoning_effort": "low"}}.
    extra_body: dict = field(default_factory=dict)


@dataclass(slots=True)
class AgentSettings:
    """Context management and the assistant's identity."""

    assistant_name: str = "KIRA"
    # Deliberately short: re-sent on every request, so each token here is paid for on
    # the first prompt-cache miss.
    system_prompt: str = (
        "You are KIRA, a local voice assistant. You are concise, warm, and practical. "
        "Answers are spoken aloud, so keep them short, avoid markdown, and do not use "
        "bullet lists. When you act with a tool, say briefly what happened."
    )
    max_context_tokens: int = 4096
    summary_trigger_tokens: int = 3200
    keep_recent_messages: int = 12
    max_turns_per_request: int = 4
    tool_timeout_s: float = 30.0
    max_tool_result_chars: int = 4000
    # When true, a tool call is returned to the LLM for a spoken summary instead of
    # being spoken verbatim.
    report_tool_results: bool = True


@dataclass(slots=True)
class LoggingSettings:
    dir: Path = DEFAULT_LOG_DIR
    level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3
    console: bool = True
    latency: bool = True  # Timestamp and report per-stage pipeline timings.


# --------------------------------------------------------------------------- #
# The settings object
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class Settings:
    audio: AudioSettings = field(default_factory=AudioSettings)
    wake: WakeSettings = field(default_factory=WakeSettings)
    stt: STTSettings = field(default_factory=STTSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    #: Directory the settings came from, used to anchor relative paths.
    config_dir: Path = field(default_factory=Path.cwd, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        self.config_dir = Path(self.config_dir)
        self._resolve_paths()
        self._validate()

    def _resolve_paths(self) -> None:
        """Turn relative paths absolute, anchored at the config file's directory."""
        base = self.config_dir
        for section_name in _PATH_FIELDS:
            section = getattr(self, section_name)
            for name in _PATH_FIELDS[section_name]:
                value = getattr(section, name)
                if isinstance(value, Path) or not isinstance(value, str):
                    continue
                setattr(section, name, _resolve(value, base))

    def _validate(self) -> None:
        a = self.audio
        _between("audio.sample_rate", a.sample_rate, 1, 96_000)
        _between("audio.channels", a.channels, 1, 15)
        _between("audio.block_ms", a.block_ms, 5, 200)
        _between("audio.wake_prebuffer_s", a.wake_prebuffer_s, 0.1, 10)
        _between("audio.vad_threshold", a.vad_threshold, 0.3, 0.8)
        _between("audio.vad_min_silence_s", a.vad_min_silence_s, 0.1, 5)
        _between("audio.vad_min_speech_s", a.vad_min_speech_s, 0.05, 10)
        _between("audio.listen_timeout_s", a.listen_timeout_s, 1, 60)
        _between("audio.tts_sample_rate", a.tts_sample_rate, 8_000, 96_000)

        w = self.wake
        if w.enabled:
            _between("wake.sensitivity", w.sensitivity, 0.0, 1.0)
            _between("wake.detection_threshold", w.detection_threshold, 0.0, 1.0)
            if not 0 < w.min_score <= 1:
                raise ValueError("wake.min_score must be in (0, 1]")

        s = self.stt
        if not s.enabled:
            return
        _between("stt.beam_size", s.beam_size, 0.5, 10)
        if s.device not in {"auto", "cpu", "cuda", "rocm", "metal"}:
            raise ValueError(f"stt.device {s.device!r} is not supported")

        t = self.tts
        if t.engine not in {"kokoro", "edge"}:
            raise ValueError(f"tts.engine {t.engine!r} is not supported")
        _between("tts.speed", t.speed, 0.5, 2.0)

        l = self.llm
        _between("llm.temperature", l.temperature, 0.0, 2.0)
        _between("llm.max_tokens", l.max_tokens, 1, 32_768)
        if not l.server_url.startswith(("http://", "https://")):
            raise ValueError(f"llm.server_url {l.server_url!r} must start with http:// or https://")

        ag = self.agent
        _between("agent.max_context_tokens", ag.max_context_tokens, 256, 1_048_576)
        if not ag.summary_trigger_tokens < ag.max_context_tokens:
            raise ValueError("agent.summary_trigger_tokens must be < max_context_tokens")
        _between("agent.keep_recent_messages", ag.keep_recent_messages, 2, 256)
        _between("agent.max_turns_per_request", ag.max_turns_per_request, 1, 32)
        _between("agent.tool_timeout_s", ag.tool_timeout_s, 1, 600)

        lg = self.logging
        if lg.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"logging.level {lg.level!r} is not a valid level")
        _between("logging.max_bytes", lg.max_bytes, 1024, 1024**3)
        _between("logging.backup_count", lg.backup_count, 0, 60)

    # -- serialization --------------------------------------------------------

    def as_dict(self) -> dict:
        """JSON-ready representation, paths as strings, in file order."""

        def walk(node):
            if isinstance(node, Path):
                return str(node)
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()}
            return node

        out = {}
        for name in _SECTION_ORDER:
            section = getattr(self, name)
            out[name] = walk(dataclasses.asdict(section))
        return out

    def describe(self) -> str:
        """Compact multi-line summary, for --check and diagnostics."""
        return "\n".join(
            [
                f"audio   {self.audio.sample_rate} Hz / {self.audio.block_ms} ms   "
                f"in={self.audio.input_device!r}  out={self.audio.output_device!r}",
                f"wake    enabled={self.wake.enabled}  sens={self.wake.sensitivity}  "
                f"thr={self.wake.detection_threshold}",
                f"stt     {self.stt.model}  device={self.stt.device}  "
                f"compute={self.stt.compute_type}  beam={self.stt.beam_size}",
                f"tts     {self.tts.engine}  voice={self.tts.kokoro_voice}  "
                f"edge={self.tts.edge_voice}",
                f"llm     {self.llm.server_url}  model={self.llm.model}  "
                f"temp={self.llm.temperature}  max={self.llm.max_tokens}",
                f"agent   {self.agent.assistant_name}  ctx={self.agent.max_context_tokens}  "
                f"sum={self.agent.summary_trigger_tokens}  turns={self.agent.max_turns_per_request}",
                f"logging {self.logging.level} -> {self.logging.dir}",
            ]
        )


_SECTION_CLASSES = {
    "audio": AudioSettings,
    "wake": WakeSettings,
    "stt": STTSettings,
    "tts": TTSSettings,
    "llm": LLMSettings,
    "agent": AgentSettings,
    "logging": LoggingSettings,
}


def _between(label: str, value, lo, hi) -> None:
    if not lo <= value <= hi:
        raise ValueError(f"{label}={value!r} must be within [{lo}, {hi}]")


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _resolve(raw: str, base: Path) -> Path:
    """Resolve a config path: absolute stays, relative anchors at `base`."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_load_warnings: list[str] = []


def get_load_warnings() -> list[str]:
    return _load_warnings


def reset_load_warnings() -> None:
    _load_warnings.clear()


def discover_config() -> Path:
    """The config path that will be used. Returns the default if none exists yet."""
    for candidate in (
        DEFAULT_CONFIG_FILE,
        PROJECT_ROOT / "kira_config.yaml",
        Path.cwd() / "kira_config.yaml",
        USER_CONFIG_DIR / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG_FILE


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML, writing a template when no config exists.

    A malformed config raises: silently substituting defaults for a user's broken file
    would hide the error and produce a wrong device or model. A *missing* config is not
    an error — defaults are written and returned.
    """
    reset_load_warnings()

    path = Path(config_path).expanduser() if config_path else discover_config()

    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_CONFIG_TEMPLATE, encoding="utf-8", newline="\n")
            get_load_warnings().append(f"created config template at {path}")
        except OSError as exc:
            get_load_warnings().append(f"could not write config to {path}: {exc}")
        return Settings(config_dir=Path.cwd())

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    if text.lstrip().startswith("["):
        get_load_warnings().append(f"{path} looks like TOML, not YAML; using defaults")
        return Settings(config_dir=path.parent)

    data = _parse_yaml(path, text)

    # An `include:` layers a shared base under a local override.
    include = data.pop("include", None)
    if include:
        inc = _resolve(str(include), path.parent)
        if not inc.exists():
            raise FileNotFoundError(f"config include target not found: {inc}")
        with inc.open(encoding="utf-8") as fh:
            base_data = _parse_yaml(inc, fh.read())
        data = {**base_data, **data}

    return build_settings(data, config_dir=path.parent)


def _parse_yaml(path: Path, text: str) -> dict:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed config at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return {k: v for k, v in data.items() if not (isinstance(k, str) and k.startswith("#"))}


def build_settings(data: dict, config_dir: Path | None = None) -> Settings:
    """Build settings from a parsed mapping.

    Unknown top-level keys are warnings, not errors: an older KIRA must open a newer
    config, and a stray hand-edit should not take the assistant down.
    """
    base = Path(config_dir) if config_dir else Path.cwd()
    kwargs = {}
    for section_name, values in data.items():
        cls = _SECTION_CLASSES.get(section_name)
        if cls is None:
            get_load_warnings().append(f"ignored unknown top-level key {section_name!r}")
            continue
        if not isinstance(values, dict):
            raise TypeError(
                f"Section {section_name!r} must be a mapping, got {type(values).__name__}"
            )
        kwargs[section_name] = _build_section(cls, values, base)

    return Settings(**kwargs, config_dir=base)


def _build_section(cls: type, values: dict, base: Path):
    """Instantiate one section, coercing values to their declared field types."""
    kwargs = {}
    for name, f in cls.__dataclass_fields__.items():
        if name not in values:
            continue
        raw = values[name]
        if f.type == "Path":
            kwargs[name] = _resolve(str(raw), base) if raw is not None else None
            continue
        kwargs[name] = _coerce_field(f, raw)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        known = set(cls.__dataclass_fields__)
        bad = sorted(set(kwargs) - known)
        hint = f"; unknown field(s): {', '.join(bad)}" if bad else ""
        raise TypeError(f"Invalid value for section {cls.__name__}{hint}: {exc}") from exc


def _coerce_field(f: dataclasses.Field, raw) -> object:
    """Coerce a YAML scalar to the field's declared type."""
    if raw is None:
        return None
    if isinstance(raw, bool) or f.type == "bool":
        return _truthy(raw)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, float):
        return raw
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text == "" or text.lower() in {"null", "~", "none"}:
            return None
        if f.type in {"int", "float"}:
            try:
                return int(text) if f.type == "int" else float(text)
            except ValueError:
                pass
        if f.type == "str":
            return text
    return raw


def apply_env_overrides(settings: Settings) -> Settings:
    """Override settings from KIRA_* environment variables; returns settings in place."""
    applied: list[str] = []
    for env, dotted in _ENV_MAP.items():
        raw = os.environ.get(env)
        if raw is None or not raw.strip():
            continue
        section_name, _, attr = dotted.partition(".")
        section = getattr(settings, section_name)
        f = dataclasses.fields(type(section))[attr]
        current = getattr(section, attr)
        new = _coerce_field(f, raw.strip())
        if new != current:
            setattr(section, attr, new)
            applied.append(f"{dotted}={raw!r}")
    if applied:
        get_load_warnings().append("env overrides: " + ", ".join(applied))
    return settings


_ENV_MAP: dict[str, str] = {
    "KIRA_LLM_SERVER_URL": "llm.server_url",
    "KIRA_LLM_BASE_URL": "llm.server_url",
    "KIRA_LLM_MODEL": "llm.model",
    "KIRA_LLM_API_KEY": "llm.api_key",
    "KIRA_INPUT_DEVICE": "audio.input_device",
    "KIRA_OUTPUT_DEVICE": "audio.output_device",
    "KIRA_SAMPLE_RATE": "audio.sample_rate",
    "KIRA_LOG_LEVEL": "logging.level",
    "KIRA_ASSISTANT_NAME": "agent.assistant_name",
    "KIRA_TTS_ENGINE": "tts.engine",
    "KIRA_STT_MODEL": "stt.model",
}


_CONFIG_TEMPLATE: str = """\
# =============================================================================
# KIRA configuration
# -----------------------------------------------------------------------------
# Every value below is the built-in default. Edit to override.
# Relative paths resolve against this file's directory.
# =============================================================================

audio:
  # input_device: null          # null = system default; int index or name substring
  # output_device: null
  sample_rate: 16000
  channels: 1
  block_ms: 40
  wake_prebuffer_s: 1.5
  vad_threshold: 0.5
  vad_min_silence_s: 0.7
  vad_min_speech_s: 0.25
  listen_timeout_s: 8.0
  echo_cancel: true
  tts_sample_rate: 24000

wake:
  enabled: true
  # model_path: null            # .onnx; null = bundled pretrained model
  sensitivity: 0.5
  detection_threshold: 0.5
  min_score: 0.25
  language_model: en_US
  rearm_silence_s: 1.2

stt:
  enabled: true
  model: base.en                # tiny.en base.en small.en medium.en large-v3
  device: auto                  # auto | cpu | cuda | rocm | metal
  compute_type: auto            # auto | int8 | int8_float16 | float16 | float32
  beam_size: 1
  vad_filter: true
  language: en

tts:
  engine: kokoro                # kokoro | edge
  kokoro_voice: af_heart
  kokoro_model: kokoro-v1.0
  speed: 1.0
  edge_voice: en-US-AriaNeural
  rate: '+0%'
  pitch: '+0Hz'
  fallback_to_edge: true
  stream: true

llm:
  server_url: http://127.0.0.1:11434/v1
  model: llama3.2:3b
  temperature: 0.4
  top_p: 0.9
  max_tokens: 512
  stream: true
  timeout_s: 60.0
  # api_key: ollama

agent:
  assistant_name: KIRA
  max_context_tokens: 4096
  summary_trigger_tokens: 3200
  keep_recent_messages: 12
  max_turns_per_request: 4
  tool_timeout_s: 30.0
  max_tool_result_chars: 4000
  report_tool_results: true

logging:
  dir: ./logs
  level: INFO
  max_bytes: 5242880
  backup_count: 3
  console: true
  latency: true
"""


if __name__ == "__main__":
    settings = load_settings()
    print("KIRA configuration")
    print("-" * 64)
    print(f"config   {discover_config()}")
    print(settings.describe())
    if _load_warnings:
        print()
        for warning in _load_warnings:
            print(f"  ! {warning}")
    print()
    print("round-trip:", "ok" if build_settings(settings.as_dict(), settings.config_dir) else "FAILED")
