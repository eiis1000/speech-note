#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe local whisper.cpp backend loading.")
    parser.add_argument("--binary", type=Path, default=None, help="Path to whisper-cli. Defaults to PATH lookup.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary or shutil.which("whisper-cli")
    payload: dict[str, object] = {
        "binary": str(binary) if binary is not None else None,
        "found": binary is not None,
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
        "vulkan_icd_dir_exists": Path("/run/opengl-driver/share/vulkan/icd.d").exists(),
        "dev_dri_exists": Path("/dev/dri").exists(),
    }
    if binary is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    result = subprocess.run(
        [str(binary), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    stderr = result.stderr
    payload.update(
        {
            "returncode": result.returncode,
            "loaded_vulkan_backend": "loaded Vulkan backend" in stderr,
            "loaded_cpu_backend": "loaded CPU backend" in stderr,
            "vulkan_no_devices": "No devices found" in stderr,
            "stderr_first_lines": stderr.splitlines()[:8],
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
