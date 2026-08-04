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


# Prompt architecture (settled by a bake-off across seven hosted models, 2026-08-04;
# the run scripts and scores are in AUDIT.md's follow-up):
#   * The whole contract lives HERE, in the system prompt; the user turns carry data
#     only. Small models keep a contract better when it is not interleaved with data.
#   * A synthetic worked example rides along as a real user/assistant message pair.
#     Pasting an example into the system prompt made models imitate its surface
#     instead of the schema (bare entries, no envelope); an assistant turn is the
#     strongest format anchor there is. The example is invented, so no real
#     transcript text ships in any prompt.
#   * The calibration names the base rate ("a difficult recording usually yields
#     several entries"). The previous closing line praised the empty list, and most
#     models under it returned zero notes on recordings with half a dozen genuine
#     divergences.
SYSTEM_PROMPT = (
    "You are a transcript auditor. You receive several machine transcripts (sources) of "
    "one recording, plus a cleaned transcript built from them. Find the places where the "
    "cleaned transcript says something the sources do not jointly support, and report "
    "them as JSON data. You never rewrite the transcript.\n\n"
    "All transcripts cover the same recording from beginning to end, in order. A "
    "disagreement is therefore LOCAL: it happens at one moment, and the competing "
    "readings are what different recognizers produced at that same moment. Find the "
    "moment by the words around it. Text from a different part of the recording is never "
    "an alternative, no matter how similar it sounds.\n\n"
    "Report an entry when:\n"
    "- The sources give different, mutually incompatible text for the same stretch of "
    "speech. The audio was hard there and each recognizer guessed; the cleaned "
    "transcript picked one guess, and the reader deserves the others.\n"
    "- Exactly one source has a fluent, specific claim (a name, a place, an event, a "
    "relationship) at a moment where the others are garbled or trail off. Confident "
    "text supported by one source alone is often invention; report it, with whatever "
    "the other sources have at that moment as the alternatives.\n\n"
    "Do NOT report:\n"
    "- Text present in one source and merely absent from the others. Recognizers differ "
    "in sensitivity; a quiet passage only one heard is normally real.\n"
    "- Wording, spelling, or punctuation differences.\n"
    "- Disfluencies the cleanup removed.\n\n"
    'Answer with exactly this JSON shape: {"uncertain": [{"quote": "...", '
    '"alternatives": ["..."]}]}\n'
    '- "quote": copied character-for-character from the CLEANED TRANSCRIPT; the '
    "shortest span that covers the doubtful claim (a clause, not a paragraph), at most "
    f"{ANNOTATION_MAX_QUOTE_CHARS} characters.\n"
    '- "alternatives": what the OTHER sources have at that same moment, cleaned up just '
    f"enough to be readable, most plausible first, at most {ANNOTATION_MAX_ALTERNATIVES}. "
    "A source that is garbled or silent at that moment contributes nothing: give fewer "
    "alternatives rather than reach elsewhere in the recording.\n"
    "- Skip an entry whose alternatives all mean the same thing as the quote.\n"
    "- Order entries as they appear in the cleaned transcript.\n\n"
    "Calibration: a clean, well-heard recording yields an empty list; a difficult "
    "recording usually yields several entries. Report every place the sources visibly "
    "diverge — withholding a real divergence misleads the reader exactly as much as "
    "inventing one."
)

_ASK = "Report the passages the sources do not jointly support."

# The worked example (synthetic on purpose). It demonstrates: several entries from one
# recording, clause-sized quotes, alternatives read off the same moment, a garbled
# source contributing nothing (recognizer-b trails off where "grilled" is disputed),
# and agreed text ("for everyone") not reported.
EXAMPLE_USER = (
    "Source 1 — recognizer-a:\n"
    "so we drove out to the lake house on friday and my brother grilled uh grilled fish "
    "for everyone\n\n"
    "Source 2 — recognizer-b:\n"
    "so we drove to the lake house on sunday and my brother uh for everyone\n\n"
    "Source 3 — recognizer-c:\n"
    "so we drove out to the lake house on friday and my brother um brought fish for "
    "everyone\n\n"
    "CLEANED TRANSCRIPT:\n"
    "We drove out to the lake house on Friday, and my brother grilled fish for everyone.\n\n"
    f"{_ASK}"
)

EXAMPLE_ASSISTANT = (
    '{"uncertain": ['
    '{"quote": "on Friday", "alternatives": ["on Sunday"]}, '
    '{"quote": "my brother grilled fish", "alternatives": ["my brother brought fish"]}'
    "]}"
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
    """The data-only user turn. All rules live in SYSTEM_PROMPT; mirroring them here
    taught nothing and diluted the contract (see the prompt-architecture note above)."""
    blocks = "\n\n".join(
        f"Source {index} — {source.label} ({source.model}):\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    return f"{blocks}\n\nCLEANED TRANSCRIPT:\n{cleaned_text}\n\n{_ASK}"


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
    for at all — distinct from a well-formed empty list, which is a real answer.

    The object is prefix-parsed (raw_decode) rather than sliced to the last brace:
    providers without schema enforcement were observed closing the envelope after the
    first entry and continuing anyway ('{"uncertain": [..]}, {..}]}'), and the valid
    prefix of such a reply is a real answer while the whole is unparseable.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start == -1:
        return [], False
    try:
        payload = json.JSONDecoder().raw_decode(text[start:])[0]
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
    estimated = sum(
        estimate_text_tokens(text)
        for text in (SYSTEM_PROMPT, EXAMPLE_USER, EXAMPLE_ASSISTANT, prompt)
    )
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
                {"role": "user", "content": EXAMPLE_USER},
                {"role": "assistant", "content": EXAMPLE_ASSISTANT},
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
