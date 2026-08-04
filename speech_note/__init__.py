"""Local speech-to-text note taker.

Capture or load audio, transcribe with a collection of ASR sources (local Whisper +
Parakeet, and optionally a remote audio-LLM), clean the transcript with a local or
remote LM, and archive every run.
"""

# Bumped one patch per commit; surfaced by `speech-note --version`.
__version__ = "2.1.24"
