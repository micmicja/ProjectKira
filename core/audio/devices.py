"""Audio device enumeration and routing.

These exist to let ``kira.py --check`` tell the user what they actually have before
anything heavy is imported.

Device selection accepts three forms everywhere:

    None            system default (or, for capture, the OS default input)
    3               a device index (see ``kira.py --list-devices``)
    "Yeti"          the first device whose name contains the substring, case-insensitive

The code supports both sounddevice 0.4.x and 0.5.x. They differ in two ways that matter
here: 0.4.x exposes ``get_host_api_count()``/``get_host_api()``, while 0.5.x exposes
``query_hostapis()``. ``_host_api_names()`` handles both so enumeration never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.log import get_logger

log = get_logger(__name__)


def _clean_name(name: str) -> str:
    r"""Normalise a device name for display.

    Windows hands names straight out of the driver, e.g.
    ``Headset (@System32\drivers\bthhfenum.sys,#2;%1 Hands-Free%0;(mike))``. The
    ``@path,#N;`` part is a Win32 format-resource reference and the ``%N`` specifiers
    are positional slots inside it, so neither is meaningful to a human. The string can
    also carry a NUL byte that makes ``print`` raise UnicodeEncodeError. Sanitising once
    here keeps every call site clean.
    """
    if not name:
        return ""

    # Control bytes (including NUL) are not encodable on the console.
    text = "".join(ch for ch in name if ch.isprintable() or ch in " \t")
    text = re.sub(r"@[^\s();,]+\.[A-Za-z]{2,4},#\d+;", "", text)  # format-resource ref
    text = re.sub(r"%\d+", " ", text)  # positional specifiers
    text = re.sub(r"[;]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # The substitutions above can leave "( " or " )" or a bare "()".
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.replace("()", "")


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """A sound device. ``index`` is the id to pass back to a stream."""

    index: int
    name: str
    subname: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    host_api: str

    def __post_init__(self) -> None:
        # Device names on Windows can carry driver-resource placeholders and control
        # bytes (@System32\drivers\..., \x00). They are unusable in a UI and can make
        # printing crash, so sanitise once here rather than at every call site.
        object.__setattr__(self, "name", _clean_name(self.name))
        object.__setattr__(self, "subname", _clean_name(self.subname))

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    @property
    def label(self) -> str:
        """The short form used in dropdowns."""
        return f"[{self.index:>2}] {self.name}" if self.name else f"[{self.index:>2}] device {self.index}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        kinds = "in " if self.is_input else "out"
        if self.is_input and self.is_output:
            kinds = "i/o"
        suffix = f" ({self.subname})" if self.subname else ""
        return f"{self.label} {kinds} {int(self.default_sample_rate)} Hz{suffix}"


def sounddevice_available() -> bool:
    """Whether the audio backend can be imported. Never raises."""
    try:
        import sounddevice  # noqa: F401

        return True
    except ImportError:
        return False


def _host_api_names() -> dict[int, str]:
    """Map host API id -> display name, tolerating 0.4.x and 0.5.x."""
    import sounddevice as sd

    if hasattr(sd, "query_hostapis"):
        out = {}
        for i, api in enumerate(sd.query_hostapis()):
            name = api.get("name") if isinstance(api, dict) else getattr(api, "name", "")
            out[i] = str(name)
        return out

    if hasattr(sd, "get_host_api_count"):
        count = int(sd.get_host_api_count())
        return {i: str(sd.get_host_api(i).name) for i in range(count)}

    return {}


def list_audio_devices() -> list[AudioDevice]:
    """Every device the backend can see, ordered by index.

    Raises :class:`RuntimeError` with the install command in the message: this path
    shows up in ``--check`` and should never fail silently.
    """
    if not sounddevice_available():
        raise RuntimeError(
            "sounddevice is not installed. Run:  pip install sounddevice\n"
            "On Windows you may also need the binary portaudio wheel:  pip install PortAudio"
        )
    import sounddevice as sd

    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - hardware dependent
        log.warning("device enumeration failed: %s", exc)
        raise RuntimeError(f"could not enumerate audio devices: {exc}") from exc

    host_names = _host_api_names()
    out: list[AudioDevice] = []
    for raw in devices:
        dev = dict(raw) if not isinstance(raw, dict) else raw
        index = int(dev.get("index", len(out)))
        out.append(AudioDevice(
            index=index,
            name=str(dev.get("name", f"device {index}")),
            subname=str(dev.get("subname", "") or ""),
            max_input_channels=int(dev.get("max_input_channels", 0) or 0),
            max_output_channels=int(dev.get("max_output_channels", 0) or 0),
            default_sample_rate=float(dev.get("default_samplerate", 0) or 0),
            host_api=host_names.get(int(dev.get("hostapi", 0) or 0), ""),
        ))
    return out


def resolve_device(
    selection: str | int | None, *, inputs_only: bool = False, outputs_only: bool = False
) -> int | None:
    """Resolve a user selection to a device index, or ``None`` for the OS default.

    Unknown indices and name substrings raise :class:`KeyError` listing the matching
    devices, so a typo in config is a clear message rather than audio going to the
    wrong sink.
    """
    if selection is None:
        return None

    devices = list_audio_devices()
    if not devices:
        raise RuntimeError("no audio devices found")

    # An int and an all-digit string both mean an index, but "0" pasted out of
    # --list-devices is a string, so normalise before branching.
    if isinstance(selection, str) and selection.strip().isdigit():
        selection = int(selection.strip())

    if isinstance(selection, int):
        matches = [d for d in devices if d.index == selection]
        if matches and _acceptable(matches[0], inputs_only, outputs_only):
            return matches[0].index
        raise KeyError(
            f"audio device index {selection} has no usable streams. Available: "
            + "; ".join(d.label for d in _acceptable_list(devices, inputs_only, outputs_only))
        )

    text = str(selection).strip()
    if text.lower() in {"", "default", "auto"}:
        return None
    needle = text.lower()
    matches = [d for d in devices if needle in d.name.lower()]
    matches = [d for d in matches if _acceptable(d, inputs_only, outputs_only)]
    if not matches:
        raise KeyError(
            f"no audio device matching {text!r}. Available: "
            + "; ".join(d.label for d in _acceptable_list(devices, inputs_only, outputs_only))
        )
    return matches[0].index


def default_input_index() -> int | None:
    """The OS default capture device, or None if there is none."""
    if not sounddevice_available():
        return None
    import sounddevice as sd

    try:
        value = sd.default.device[0]
    except Exception:  # pragma: no cover
        return None
    return None if value in (-1, "none", None) else int(value)


def default_output_index() -> int | None:
    """The OS default playback device, or None if there is none."""
    if not sounddevice_available():
        return None
    import sounddevice as sd

    try:
        value = sd.default.device[1]
    except Exception:  # pragma: no cover
        return None
    return None if value in (-1, "none", None) else int(value)


def describe_devices() -> str:
    """A human-readable block for ``--check`` output."""
    if not sounddevice_available():
        return "  (unavailable: sounddevice is not installed — pip install sounddevice)"
    try:
        devices = list_audio_devices()
    except RuntimeError as exc:
        return f"  (unavailable: {exc})"
    if not devices:
        return "  (no audio devices found)"

    lines = [f"  defaults: in={default_input_index()}  out={default_output_index()}"]
    for dev in devices:
        if not dev.is_input and not dev.is_output:
            continue
        lines.append(f"  {dev}")
    return "\n".join(lines)


def device_cache_path() -> Path:
    """Where the routing snapshot is persisted between runs."""
    from core import config as _config

    return _config.PROJECT_DATA_DIR / "audio_devices.json"


# --------------------------------------------------------------------------- #

def _acceptable(dev: AudioDevice, inputs_only: bool, outputs_only: bool) -> bool:
    if inputs_only and not dev.is_input:
        return False
    if outputs_only and not dev.is_output:
        return False
    return dev.is_input or dev.is_output


def _acceptable_list(devices: list[AudioDevice], inputs_only: bool, outputs_only: bool) -> list[AudioDevice]:
    return [d for d in devices if _acceptable(d, inputs_only, outputs_only)]