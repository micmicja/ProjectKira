"""KIRA — local-first voice assistant and agent.

    python kira.py               run the assistant (GUI when PyQt6 is present)
    python kira.py --check       validate config, report devices and dependencies
    python kira.py --list-devices
    python kira.py --chat        terminal agent loop, no GUI, no audio
    python kira.py --verbose --no-window
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Qt reads this before any QApplication is constructed.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from core.config import Settings, apply_env_overrides, get_load_warnings, load_settings  # noqa: E402
from core.log import get_logger, setup_logging  # noqa: E402

# Distribution -> import name, for dependency reporting.
_DIST_IMPORTS = {
    "pyqt6": "PyQt6",
    "numpy": "numpy",
    "pyyaml": "yaml",
    "openai": "openai",
    "openwakeword": "openwakeword",
    "faster-whisper": "faster_whisper",
    "kokoro": "kokoro",
    "edge-tts": "edge_tts",
    "sounddevice": "sounddevice",
    "torch": "torch",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kira",
        description="KIRA — local-first voice assistant and agent.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a config YAML (default: data/kira.yaml, created if missing).",
    )
    parser.add_argument("--server", dest="override_server_url", default=None,
                        help="Override the LLM endpoint, e.g. http://127.0.0.1:11434/v1")
    parser.add_argument("--model", dest="override_model", default=None,
                        help="Override the model identifier, e.g. llama3.2:3b")
    parser.add_argument("--mic", dest="override_input_device", default=None,
                        help="Capture device: index or name substring")
    parser.add_argument("--speaker", dest="override_output_device", default=None,
                        help="Playback device: index or name substring")
    parser.add_argument("--log-dir", dest="override_log_dir", type=Path, default=None,
                        help="Directory for kira.log (default: data/logs)")
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG")
    parser.add_argument("--no-window", action="store_true", help="Run without the PyQt6 window")
    parser.add_argument("--chat", action="store_true", help="Terminal agent loop")
    parser.add_argument("--check", action="store_true", help="Report configuration and exit")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--version", action="version", version="kira 0.1.0")
    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    """CLI flags win over the YAML file; the user's explicit intent wins everywhere."""
    if args.override_server_url:
        settings.llm.server_url = args.override_server_url
    if args.override_model:
        settings.llm.model = args.override_model
    if args.override_input_device is not None:
        settings.audio.input_device = args.override_input_device
    if args.override_output_device is not None:
        settings.audio.output_device = args.override_output_device
    if args.override_log_dir is not None:
        settings.logging.dir = args.override_log_dir.resolve()
    if args.verbose:
        settings.logging.level = "DEBUG"


# --------------------------------------------------------------------------- #
# --check
# --------------------------------------------------------------------------- #

def _installed(dist: str) -> str:
    module = _DIST_IMPORTS.get(dist, dist.replace("-", "_"))
    try:
        mod = __import__(module)
    except ImportError:
        return "not installed"
    return getattr(mod, "__version__", "ok")


def check_environment(settings: Settings) -> int:
    """Print a config/hardware report. Returns a process exit code."""
    from core.llm import LLMClient

    lines: list[str] = []
    lines.append("KIRA configuration")
    lines.append("=" * 68)
    lines.append(f"  config file   {discover_config_path(settings)}")
    lines.append("")
    lines.append(settings.describe().replace("\n", "\n"))
    lines.append("")

    llm = LLMClient(settings.llm)
    lines.append(f"  llm client    {'importable' if llm.is_available else 'openai package MISSING'}")
    if llm.is_available:
        reachable = llm.ping(timeout_s=2.0)
        lines.append(f"  llm endpoint  {'reachable' if reachable else 'NOT REACHABLE'}")
        if reachable:
            models = llm.list_models()
            if models:
                lines.append(f"  models on   {', '.join(models[:6])}{' …' if len(models) > 6 else ''}")
                if settings.llm.model not in models:
                    lines.append(f"  ! configured model {settings.llm.model!r} not in that list")

    lines.append("")
    lines.append("Audio devices")
    lines.append("-" * 68)
    try:
        from core.audio.devices import describe_devices

        lines.append(describe_devices())
    except RuntimeError as exc:
        lines.append(f"  unavailable: {exc}")

    lines.append("")
    lines.append("Dependencies")
    lines.append("-" * 68)
    lines.append(f"  python     {sys.version.split()[0]}")
    for dist, label in (
        ("pyqt6", "pyqt6"),
        ("numpy", "numpy"),
        ("pyyaml", "pyyaml"),
        ("openai", "openai"),
        ("sounddevice", "sounddevice"),
        ("openwakeword", "openwakeword"),
        ("faster-whisper", "faster_whisper"),
        ("kokoro", "kokoro"),
        ("edge-tts", "edge_tts"),
        ("torch", "torch"),
    ):
        lines.append(f"  {label:<15}{_installed(dist)}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            version = subprocess.check_output(
                ["ffmpeg", "-version"], text=True, timeout=5, stderr=subprocess.DEVNULL
            ).splitlines()[0]
        except Exception:
            version = "(unversioned)"
        lines.append(f"  ffmpeg         {ffmpeg}  {version}")
    else:
        lines.append("  ffmpeg         not found on PATH")

    print("\n".join(lines))

    warnings = get_load_warnings()
    if warnings:
        print()
        for warning in warnings:
            print(f"  ! {warning}")

    missing_core = [d for d in ("numpy",) if "not installed" in _installed(d)]
    if missing_core:
        print(f"\nMissing: {', '.join(missing_core)}. Run:  python -m pip install -r requirements.txt")
        return 1
    print("\nConfiguration is valid.")
    return 0


def discover_config_path(settings: Settings) -> Path:
    return settings.config_dir / "kira.yaml" if settings.config_dir else Path("data/kira.yaml")


# --------------------------------------------------------------------------- #
# --list-devices
# --------------------------------------------------------------------------- #

def list_devices() -> int:
    from core.audio.devices import list_audio_devices

    print("Audio devices (index: name)")
    for device in list_audio_devices():
        kind = "in " if device.is_input else "out"
        if not device.is_input and not device.is_output:
            continue
        print(f"  [{device.index:>2}] {kind}  {device.name}")
    return 0


# --------------------------------------------------------------------------- #
# --chat
# --------------------------------------------------------------------------- #

def run_chat(settings: Settings) -> int:
    """A terminal agent loop: no GUI, no audio, just the model and its tools."""
    from core.agents.coordinator import ConversationCoordinator
    from core.llm import LLMClient
    from core.tools.registry import Registry
    from core.tools.builtin import register_builtins

    llm = LLMClient(settings.llm)
    if not llm.is_available:
        print("The 'openai' package is required. Run:  pip install openai")
        return 1

    registry = Registry(default_timeout_s=settings.agent.tool_timeout_s)
    register_builtins(registry)

    coordinator = ConversationCoordinator(
        llm,
        registry=registry,
        settings=settings,
    )

    print(f"KIRA {settings.agent.assistant_name} — {settings.llm.model} @ {settings.llm.server_url}")
    print(f"Tools: {', '.join(registry.names()) or 'none'}")
    print("Type a message. Blank line or 'exit' to quit.\n")

    try:
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                break
            if not user or user.lower() in {"exit", "quit", "q"}:
                break
            try:
                reply = coordinator.chat(user)
            except KeyboardInterrupt:
                print()
                continue
            print(f"{settings.agent.assistant_name.lower()}> {reply}\n")
    except KeyboardInterrupt:
        pass
    return 0


# --------------------------------------------------------------------------- #
# --window
# --------------------------------------------------------------------------- #

def run_window(settings: Settings) -> int:
    """Boot the PyQt6 window. Any failure here returns an exit code, never a traceback."""
    try:
        from ui.main_window import run_gui
    except ImportError as exc:
        print(f"PyQt6 is not installed ({exc}). Run:  python -m pip install -r requirements.txt")
        return 1
    return run_gui(settings)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    settings = apply_env_overrides(settings)
    apply_overrides(settings, args)

    setup_logging(
        level=settings.logging.level,
        log_dir=settings.logging.dir if not args.check else None,
        console=not args.check,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )
    log = get_logger("kira")

    if args.check:
        return check_environment(settings)
    if args.list_devices:
        return list_devices()
    if args.chat or args.no_window:
        return run_chat(settings)

    log.info("starting", extra={"llm": settings.llm.server_url, "model": settings.llm.model})
    return run_window(settings)


if __name__ == "__main__":
    sys.exit(main())
