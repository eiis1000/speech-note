"""Uncertainty annotation: mark the claims the ASR sources do not jointly support.

A separate pass from cleanup, on purpose. Asking one model to preserve content, strip
disfluencies *and* judge its own confidence gives it interpretive leeway, and with leeway
it starts summarizing — the one thing this pipeline must never do. So the cleanup prompt
does only cleanup, and this pass audits the result.

Two properties make it safe to run over a finished transcript:

  * It returns DATA, never prose. The reply is a JSON list of spans; the markers are
    inserted by code here. The pass cannot rewrite, shorten, or reorder anything.
  * It only ever ADDS. A span it cannot place in the transcript is reported in a trailing
    section rather than dropped, and any failure at all leaves the transcript exactly as
    cleanup produced it.

The reply is constrained by a JSON schema (see config.catalog.ANNOTATION_MAX_NOTES for
why that is a correctness measure and not a nicety).
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import TYPE_CHECKING, Any

from .config import (
    ANNOTATION_MAX_ALTERNATIVES,
    ANNOTATION_MAX_NOTES,
    ANNOTATION_MAX_OUTPUT_TOKENS,
    ANNOTATION_MAX_QUOTE_CHARS,
    ORGANIZER_CONTEXT_SAFETY,
)
from .chat import request_timeout_seconds
from .textproc import estimate_text_tokens

if TYPE_CHECKING:
    from .chat import ChatClient
    from .model import Transcript


SYSTEM_PROMPT = (
    "You are a transcript auditor. You are given several machine transcripts of one "
    "recording and a cleaned transcript built from them. Your job is to find the places "
    "where the cleaned transcript states something the recording does not actually "
    "support, and to report them as DATA. You never rewrite the transcript.\n\n"
    "What counts as unsupported:\n"
    "- The sources carry DIFFERENT, mutually incompatible text for the same stretch of "
    "speech. That means the audio was unintelligible there and each recognizer guessed. "
    "The cleaned transcript picked one guess; the reader deserves to know the others.\n"
    "- One source states something fluent and specific (a name, a place, an event, a "
    "relationship) where the other sources are garbled or say something unrelated. A "
    "smooth, confident sentence supported by exactly one source is usually invention.\n\n"
    "What does NOT count:\n"
    "- Content present in one source and simply ABSENT from the others. Recognizers "
    "differ in sensitivity; a quiet passage only one heard is normally real. Leave it.\n"
    "- Ordinary wording, spelling or punctuation differences between sources.\n"
    "- Disfluencies the cleanup correctly removed.\n\n"
    "Report only genuine cases. An empty list is the correct answer for a clear "
    "recording, and is much better than padding the list with weak guesses."
)


# The schema is sent as response_format, so the model is *constrained* to it rather than
# merely asked. maxLength/maxItems are what bound the worst-case reply length.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "uncertain": {
            "type": "array",
            "maxItems": ANNOTATION_MAX_NOTES,
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string", "maxLength": ANNOTATION_MAX_QUOTE_CHARS},
                    "alternatives": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": ANNOTATION_MAX_ALTERNATIVES,
                        "items": {
                            "type": "string",
                            "maxLength": ANNOTATION_MAX_QUOTE_CHARS,
                        },
                    },
                },
                "required": ["quote", "alternatives"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["uncertain"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "uncertain_passages", "strict": True, "schema": RESPONSE_SCHEMA},
}


def build_prompt(cleaned_text: str, sources: "list[Transcript]") -> str:
    blocks = "\n\n".join(
        f"Source {index} — {source.label} ({source.model}):\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    return (
        "Below are the source transcripts and the cleaned transcript built from them.\n\n"
        f"{blocks}\n\n"
        f"CLEANED TRANSCRIPT:\n{cleaned_text}\n\n"
        "Report the passages the sources do not jointly support.\n"
        "Rules for each entry:\n"
        '- "quote" MUST be copied CHARACTER-FOR-CHARACTER from the CLEANED TRANSCRIPT '
        "above, and MUST be the shortest span that covers the doubtful claim (a clause or "
        f"a sentence, not a whole paragraph), at most {ANNOTATION_MAX_QUOTE_CHARS} "
        "characters.\n"
        '- "alternatives" are the competing readings from the other sources, cleaned up '
        f"just enough to be readable, most plausible first, at most "
        f"{ANNOTATION_MAX_ALTERNATIVES}. Do not invent alternatives: take them from what "
        "the sources actually say.\n"
        "- Do not include an entry whose alternatives all mean the same thing as the quote.\n"
        "- Order entries as they appear in the cleaned transcript.\n"
        "Return an empty list if every claim is corroborated."
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyNote:
    quote: str
    alternatives: tuple[str, ...]

    def inline_marker(self) -> str:
        joined = " / ".join(f'"{alt}"' for alt in self.alternatives)
        return f" [unclear audio; also heard as {joined}]"

    def listed(self) -> str:
        joined = " / ".join(f'"{alt}"' for alt in self.alternatives)
        return f'- "{self.quote}" — also heard as {joined}'


APPENDIX_HEADING = "Uncertain passages (the recording does not clearly support these):"


@dataclasses.dataclass
class AnnotationResult:
    """What the pass did. ``text`` is the transcript, annotated or untouched."""

    text: str
    inline_count: int = 0
    appended_count: int = 0
    error: str | None = None

    @property
    def total(self) -> int:
        return self.inline_count + self.appended_count


def _comparable(text: str) -> str:
    """Loose form for deciding whether two readings actually differ."""
    return re.sub(r"[^a-z0-9 ]+", "", " ".join(text.lower().split()))


def parse_notes(payload: object) -> list[UncertaintyNote]:
    """Notes from a decoded response body. Anything unexpected is skipped, not raised.

    Alternatives that say the same thing as the quote (or as each other) are dropped, and
    a note left with none is dropped with them. The prompt asks for this, but a weaker
    model still returns entries whose "alternative" is the quote reworded or verbatim, and
    a marker offering the reader the same words twice is worse than no marker.
    """
    entries = payload.get("uncertain") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    notes: list[UncertaintyNote] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        quote = entry.get("quote")
        raw = entry.get("alternatives")
        if not isinstance(quote, str) or not quote.strip():
            continue
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            continue
        quote = quote.strip()
        seen = {_comparable(quote)}
        alternatives: list[str] = []
        for candidate in raw:
            text = " ".join(str(candidate).split())
            key = _comparable(text)
            if not key or key in seen:
                continue
            seen.add(key)
            alternatives.append(text)
            if len(alternatives) == ANNOTATION_MAX_ALTERNATIVES:
                break
        if not alternatives:
            continue
        notes.append(UncertaintyNote(quote, tuple(alternatives)))
    return notes


def decode_response(content: str) -> tuple[list[UncertaintyNote], bool]:
    """(notes, understood). ``understood`` False means the reply was not the JSON we asked
    for at all — distinct from a well-formed empty list, which is a real answer."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return [], False
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return [], False
    if not isinstance(payload, dict) or not isinstance(payload.get("uncertain"), list):
        return [], False
    return parse_notes(payload), True


def _locate(text: str, quote: str) -> tuple[int, int] | None:
    """Find a quote, tolerating whitespace differences only.

    Nothing fuzzier on purpose: a marker attached to the wrong sentence is worse than one
    listed separately, and an unlocatable quote is never lost — it goes to the appendix.
    """
    index = text.find(quote)
    if index != -1:
        return index, index + len(quote)
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(word) for word in words), text)
    return (match.start(), match.end()) if match else None


def apply_notes(text: str, notes: list[UncertaintyNote]) -> AnnotationResult:
    """Insert markers after each located quote; list the rest in a trailing section.

    The transcript body is never altered — markers are only inserted — so this cannot
    lose content even if the auditor misbehaves.
    """
    placed: list[tuple[int, int, str]] = []
    unplaced: list[UncertaintyNote] = []
    for note in notes:
        span = _locate(text, note.quote)
        if span is None:
            unplaced.append(note)
            continue
        start, end = span
        if any(start < other_end and other_start < end for other_start, other_end, _ in placed):
            # Overlapping spans would nest markers inside each other; the first wins and
            # the second is still reported, just in the appendix.
            unplaced.append(note)
            continue
        placed.append((start, end, note.inline_marker()))

    annotated = text
    for start, end, marker in sorted(placed, reverse=True):
        annotated = annotated[:end] + marker + annotated[end:]
    if unplaced:
        listed = "\n".join(note.listed() for note in unplaced)
        annotated = f"{annotated.rstrip()}\n\n{APPENDIX_HEADING}\n{listed}"
    return AnnotationResult(
        text=annotated, inline_count=len(placed), appended_count=len(unplaced)
    )


def annotate(
    client: "ChatClient",
    *,
    cleaned_text: str,
    sources: "list[Transcript]",
    context_tokens: int,
) -> AnnotationResult:
    """Run the audit pass. Never raises: on any failure the transcript comes back as-is."""
    prompt = build_prompt(cleaned_text, sources)
    estimated = estimate_text_tokens(SYSTEM_PROMPT) + estimate_text_tokens(prompt)
    budget = max(256, int(context_tokens * ORGANIZER_CONTEXT_SAFETY))
    requested = min(ANNOTATION_MAX_OUTPUT_TOKENS, max(1024, budget - estimated))
    if estimated + requested > budget:
        return AnnotationResult(
            text=cleaned_text,
            error=(
                f"uncertainty annotation skipped: prompt ~{estimated} tokens leaves no "
                f"room in the context budget {budget}"
            ),
        )
    try:
        response = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=requested,
            timeout=request_timeout_seconds(
                configured_timeout=client.timeout,
                estimated_prompt_tokens=estimated,
                requested_output_tokens=requested,
            ),
            response_format=RESPONSE_FORMAT,
        )
    except Exception as exc:  # noqa: BLE001 — annotation must never fail the run
        return AnnotationResult(text=cleaned_text, error=f"uncertainty annotation failed: {exc}")

    if response.finish_reason == "length":
        # Should be unreachable: the schema bounds the reply. If it happens, say so
        # plainly rather than trying to piece together a half-written answer.
        return AnnotationResult(
            text=cleaned_text,
            error=(
                "uncertainty annotation hit the output limit despite the bounded schema "
                f"({requested} tokens); no notes applied"
            ),
        )
    notes, understood = decode_response(response.content)
    if not understood:
        snippet = " ".join(response.content.split())[:160]
        return AnnotationResult(
            text=cleaned_text,
            error=(
                "uncertainty annotation did not return the requested JSON "
                f"(model {response.served_model}): {snippet}"
            ),
        )
    if not notes:
        return AnnotationResult(text=cleaned_text)
    return apply_notes(cleaned_text, notes)
