"""Host GPU detection.

This project is built and tuned for an AMD *integrated* GPU (Vulkan for
whisper.cpp / llama.cpp, ROCm for PyTorch, a CPU secondary ASR pass that overlaps
the GPU primary). It still runs anywhere, but on a discrete or NVIDIA GPU the
defaults leave performance on the table — see the "Hardware notes and other GPUs"
section of the README. We detect that case once at startup and print a pointer.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def detect_unsupported_gpu() -> str | None:
    """Return a short description of an NVIDIA/discrete GPU if one looks present.

    Returns None when nothing beyond a single (presumably integrated) GPU is
    found. Best-effort and side-effect free: every probe is guarded, so a missing
    tool or an unreadable path just means "no signal", never an error.
    """
    # NVIDIA is the strongest, cheapest signal.
    if shutil.which("nvidia-smi") or Path("/proc/driver/nvidia/version").exists():
        return "an NVIDIA GPU"
    if glob.glob("/dev/nvidia[0-9]*"):
        return "an NVIDIA GPU"

    # Otherwise ask lspci about display controllers. On a laptop the integrated
    # GPU shows up as "VGA compatible controller"; a discrete card typically adds
    # a separate "3D controller" line (or a second VGA controller).
    lspci = shutil.which("lspci")
    if lspci is None:
        return None
    try:
        output = subprocess.run(
            [lspci], check=False, capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return None
    gpu_lines = [
        line
        for line in output.splitlines()
        if any(tag in line for tag in ("VGA compatible controller", "3D controller", "Display controller"))
    ]
    if any("NVIDIA" in line for line in gpu_lines):
        return "an NVIDIA GPU"
    if any("3D controller" in line for line in gpu_lines):
        return "a discrete GPU"
    if len(gpu_lines) >= 2:
        return "multiple GPUs (a discrete GPU?)"
    return None


def warn_on_unsupported_gpu(out=sys.stderr) -> None:
    """Print a one-time, non-fatal pointer when an unsupported GPU is detected.

    Suppress with SPEECH_NOTE_NO_GPU_WARNING=1.
    """
    if os.environ.get("SPEECH_NOTE_NO_GPU_WARNING") == "1":
        return
    found = detect_unsupported_gpu()
    if found is None:
        return
    print(
        f"warning: detected {found}. speech-note is configured for an AMD integrated GPU "
        "(Vulkan for Whisper/cleanup, ROCm for PyTorch, a CPU secondary ASR pass) and is "
        "not set up for discrete or NVIDIA GPUs. It will still run, but see the "
        '"Hardware notes and other GPUs" section of the README to retarget it and get '
        "the most out of your hardware. (Silence this with SPEECH_NOTE_NO_GPU_WARNING=1.)",
        file=out,
    )
