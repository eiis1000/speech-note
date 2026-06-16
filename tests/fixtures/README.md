# Test fixtures

The full-stack suite (`tests/test_e2e.py`, gated by `SPEECH_NOTE_E2E=1`) runs the
real pipeline on real audio. It is **not** shipped with recordings — provide your
own:

- `short.wav` — ~15 s of clear English speech
- `long.wav` — ~50 s of clear English speech (looped internally past the model's
  ~400 s position limit to exercise long-form transcription)

Any mono/stereo WAV that ffmpeg can read works; the tests normalize and resample.
Without these files the e2e tests self-skip.
