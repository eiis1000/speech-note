"""Local speech-to-text note taker.

Capture or load audio, transcribe with Whisper plus an optional secondary ASR
model, clean the transcript with a local or remote LM, and archive every run.
"""

# Bumped one patch per commit; surfaced by `speech-note --version`.
__version__ = "2.1.0"
