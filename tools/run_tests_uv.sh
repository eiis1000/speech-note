#!/usr/bin/env bash
# Run the test suite WITHOUT the Nix dev shell — PyPI-only, for when the Nix binary
# cache is unreachable. Live-capture tests that need real audio hardware skip; the
# other ~190 run. The Nix dev shell remains the authoritative environment: run the
# suite there too once the cache is reachable again.
#
# setuptools<81 is pinned because webrtcvad imports pkg_resources, which newer
# setuptools removed.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-project \
  --with numpy --with requests --with webrtcvad --with 'setuptools<81' \
  python -m unittest discover -s tests "$@"
