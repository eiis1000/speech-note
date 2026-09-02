#!/usr/bin/env bash
# Run the logic suite WITHOUT the Nix dev shell — PyPI wheels only, for when the Nix
# binary cache is unreachable. The Nix dev shell remains the authoritative
# environment: run the suite there too once the cache is reachable again.
#
# Only numpy and requests are needed. sounddevice/PortAudio and webrtcvad are
# imported lazily (see devices.require_sounddevice / capture.require_webrtcvad), so
# the tests that need them self-skip rather than failing to import — pass
# --with webrtcvad --with 'setuptools<81' to run those five too.
#
# The interpreter is pinned: uv's default CPython 3.14.3 build cannot write a zip
# member (struct.error out of zipfile.FileHeader), which fails the two archive tests
# for reasons that have nothing to do with this project.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-project --python 3.13 \
  --with numpy --with requests \
  python -m unittest discover -s tests "$@"
