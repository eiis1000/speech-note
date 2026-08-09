"""Input device discovery and selection."""

from __future__ import annotations

from typing import Any, cast

# sounddevice needs the PortAudio C library, which only live capture actually uses.
# Importing it lazily keeps file-input runs (and the test suite) working on machines
# without the audio stack; the functions that really need it raise at call time.
try:
    import sounddevice as _sounddevice
except (ImportError, OSError) as exc:  # OSError: module present, PortAudio missing
    _sounddevice = None  # type: ignore[assignment]
    _SOUNDDEVICE_ERROR: Exception | None = exc
else:
    _SOUNDDEVICE_ERROR = None

from .config import CHANNELS, VAD_SUPPORTED_SAMPLE_RATES
from .terminal import read_single_choice


def require_sounddevice() -> Any:
    """The sounddevice module, or a clear error where live capture is attempted."""
    if _sounddevice is None:
        raise RuntimeError(
            "live audio capture needs sounddevice/PortAudio (available in the Nix dev "
            "shell); file and text input still work without it"
        ) from _SOUNDDEVICE_ERROR
    return _sounddevice


def coerce_input_device(device: object) -> int | str | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    text = str(device).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return text


def list_input_devices() -> list[dict[str, object]]:
    sd = require_sounddevice()
    devices = cast("list[dict[str, Any]]", sd.query_devices())
    default_input = sd.default.device[0]
    entries: list[dict[str, object]] = []
    for index, device in enumerate(devices):
        if device["max_input_channels"] <= 0:
            continue
        entries.append(
            {
                "index": index,
                "name": device["name"],
                "inputs": device["max_input_channels"],
                "default_samplerate": device["default_samplerate"],
                "is_default": index == default_input,
            }
        )
    return entries


def print_input_devices() -> None:
    for entry in list_input_devices():
        default_marker = " default" if entry["is_default"] else ""
        print(
            f"{entry['index']}: {entry['name']} "
            f"(inputs={entry['inputs']}, default_sr={entry['default_samplerate']}){default_marker}"
        )


def prompt_ordered_input_devices(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    def score(entry: dict[str, object]) -> tuple[int, str]:
        name = str(entry["name"]).lower()
        if entry["is_default"]:
            return (0, name)
        if name == "default":
            return (1, name)
        if name == "pipewire":
            return (2, name)
        return (3, name)

    return sorted(entries, key=score)


def make_hotkeys(count: int) -> list[str]:
    alphabet = "123456789abcdefghijklmnopqrstuvwxyz"
    if count > len(alphabet):
        raise SystemExit("Too many input devices to assign one-key shortcuts.")
    return list(alphabet[:count])


def select_input_device() -> int | str | None:
    entries = list_input_devices()
    if not entries:
        raise SystemExit("No input devices found.")
    ordered = prompt_ordered_input_devices(entries)
    hotkeys = make_hotkeys(len(ordered))
    key_to_entry = dict(zip(hotkeys, ordered, strict=True))
    print("Select your mic:")
    for key, entry in zip(hotkeys, ordered, strict=True):
        marker = ""
        if entry["is_default"] or str(entry["name"]).lower() == "default":
            marker = " [recommended]"
        elif str(entry["name"]).lower() == "pipewire":
            marker = " [fallback]"
        print(
            f"[{key}] {entry['name']} "
            f"(device {entry['index']}, inputs={entry['inputs']}, sr={entry['default_samplerate']}){marker}"
        )
    choice = read_single_choice(
        set(hotkeys),
        "Press key for mic. Enter selects default. Ctrl-C cancels.",
        default_key=hotkeys[0],
    )
    return coerce_input_device(key_to_entry[choice]["index"])


def choose_live_capture_sample_rate(device: object, preferred_rate: int) -> int:
    """Pick a capture rate the device supports and webrtcvad accepts."""
    sd = require_sounddevice()
    try:
        device_info = cast("dict[str, Any]", sd.query_devices(device, "input"))
        default_rate = int(round(float(device_info["default_samplerate"])))
    except Exception:
        default_rate = preferred_rate
    candidates: list[int] = []
    if default_rate in VAD_SUPPORTED_SAMPLE_RATES:
        candidates.append(default_rate)
    for rate in VAD_SUPPORTED_SAMPLE_RATES:
        if rate not in candidates:
            candidates.append(rate)
    if preferred_rate not in candidates:
        candidates.append(preferred_rate)
    for rate in candidates:
        try:
            sd.check_input_settings(
                device=device,
                samplerate=rate,
                channels=CHANNELS,
                dtype="int16",
            )
            return rate
        except Exception:
            continue
    return preferred_rate


def choose_replay_capture_sample_rate(preferred_rate: int) -> int:
    if preferred_rate in VAD_SUPPORTED_SAMPLE_RATES:
        return preferred_rate
    return 16_000
