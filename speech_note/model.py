"""Domain types.

Transcripts carry their provenance everywhere; outcomes are values, not
string-prefixed error messages.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Transcript:
    """A transcript with provenance.

    label:   stable identity of this source within the run ("asr1".."asrN" in
             collection order, "live", "extra:<filename>", "user").
    model:   identifier of whatever produced it (ASR model name, "human", ...).
    kind:    "asr-final" | "asr-live" | "external" | "user".
    quality_hint: optional one-line description of known reliability, passed to
             the cleanup LM so it can weigh sources by what we actually know.
    """

    label: str
    model: str
    kind: str
    text: str
    quality_hint: str | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclasses.dataclass
class AsrOutcome:
    """Result of one ASR pass: exactly one of transcript / skip / error."""

    name: str
    transcript: Transcript | None = None
    skip_reason: str | None = None
    error: str | None = None
    seconds: float | None = None
    realtime_factor: float | None = None

    @property
    def ok(self) -> bool:
        return self.transcript is not None


@dataclasses.dataclass
class CleanupOutcome:
    """Result of the cleanup stage."""

    text: str = ""
    method: str = "off"  # off | heuristic | llama | skipped-too-large | error | empty
    served_model: str | None = None
    finish_reason: str | None = None
    flagged_short: bool = False
    error: str | None = None
    warning: str | None = None
    estimated_prompt_tokens: int | None = None
    requested_output_tokens: int | None = None
    request_timeout: float | None = None
    seconds: float | None = None
    # Uncertainty annotation (a second, separate pass). ``text`` above is the final
    # output *including* any anchors, so every writer stays unchanged; this keeps the
    # pre-annotation text for diagnostics and records how the pass went. Annotation is
    # additive and best-effort: annotation_error never fails the run.
    text_before_annotation: str | None = None
    annotation_count: int = 0
    # Of those, how many appear in the notes list without a body anchor because their
    # quote could not be located verbatim. They are reported, never dropped.
    annotation_appendix_count: int = 0
    # Alternatives whose citation matched no source text: dropped and counted, so the
    # diagnostics show when a model fabricates readings (see annotate.verify_notes).
    annotation_rejected_citations: int = 0
    annotation_seconds: float | None = None
    # The structured notes as applied (quote/alternatives/anchored), so post-hoc
    # analysis never has to parse anchors back out of the rendered text.
    annotation_notes: list[dict[str, object]] | None = None
    annotation_error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text) and self.error is None
