"""Configuration facade.

Split into three cohesive modules, re-exported here so ``from .config import X``
keeps working unchanged:
  catalog  — static defaults, model tables, BackendSpec/AsrSource
  parsing  — --asr spec parsing and the user asr/env files
  display  — status-line short names and cleanup-prompt hints
"""

from __future__ import annotations

from .catalog import *  # noqa: F401,F403
from .display import *  # noqa: F401,F403
from .parsing import *  # noqa: F401,F403
