"""Uncertainty annotation: mark the claims the ASR sources do not jointly support.

A separate pass from cleanup, on purpose. Asking one model to preserve content, strip
disfluencies *and* judge its own confidence gives it interpretive leeway, and with leeway
it starts summarizing — the one thing this pipeline must never do. So the cleanup prompt
does only cleanup, and this pass audits the result.

Two properties make it safe to run over a finished transcript:

  * It returns DATA, never prose. The reply is a JSON list of spans; the anchors are
    inserted by code here. The pass cannot rewrite, shorten, or reorder anything.
  * It only ever ADDS. Every note lands in one numbered list at the end of the
    transcript; located notes also get a [n] anchor in the body. A quote that cannot
    be placed is still listed — never dropped — and any failure at all leaves the
    transcript exactly as cleanup produced it.

The rendering is footnote-style on purpose: an earlier inline form spelled out every
alternative inside the prose, and at realistic density (7 notes on a 3-paragraph
recording) the markers swamped the text they were annotating.

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
    ANNOTATION_MAX_CONTEXT_CHARS,
    ANNOTATION_MAX_NOTES,
    ANNOTATION_MAX_NOTES_CEILING,
    ANNOTATION_MAX_OUTPUT_TOKENS,
    ANNOTATION_MAX_QUOTE_CHARS,
    ANNOTATION_MAX_VERBATIM_CHARS,
    ANNOTATION_TOKENS_PER_NOTE,
    ANNOTATION_WORDS_PER_NOTE,
    ORGANIZER_CONTEXT_SAFETY,
)
from .chat import request_timeout_seconds
from .textproc import estimate_text_tokens

if TYPE_CHECKING:
    from .chat import ChatClient
    from .model import Transcript


# Prompt architecture (settled by a bake-off across seven hosted models, 2026-08-04).
# The bake-off is reproducible: tools/annotation_eval.py imports everything below and
# scores it against the labeled cases in evals/, so a prompt change gets a number
# before it ships. Three findings, all of which cost recall when ignored:
#   * The whole contract lives HERE, in the system prompt; the user turns carry data
#     only. Small models keep a contract better when it is not interleaved with data.
#   * A synthetic worked example rides along as a real user/assistant message pair.
#     Pasting an example into the system prompt made models imitate its surface
#     instead of the schema (bare entries, no envelope); an assistant turn is the
#     strongest format anchor there is. The example is invented, so no real
#     transcript text ships in any prompt.
#   * Calibration is semantic, not numerical. Giving the model a target number of
#     entries makes lexical noise look like evidence on difficult recordings.
SYSTEM_PROMPT = (
    "You are a transcript auditor. You receive several machine transcripts (sources) of "
    "one recording, plus a cleaned transcript built from them. Find places where a source "
    "offers a coherent, plausible meaning that materially contradicts the meaning chosen "
    "by the cleaned transcript, and report them as JSON data. You never rewrite the "
    "transcript.\n\n"
    "All transcripts cover the same recording from beginning to end, in order. A "
    "disagreement is therefore LOCAL: it happens at one moment, and the competing "
    "readings are what different recognizers produced at that same moment. Find the "
    "moment by the words around it. Text from a different part of the recording is never "
    "an alternative, no matter how similar it sounds.\n\n"
    "Report an entry only when the sources give different, mutually incompatible "
    "MEANINGS for the same stretch of speech. An alternative must itself be a coherent, "
    "plausible interpretation of that moment, such as a different name, number, date, "
    "action, object, or negation. Different words are not an alternative when they "
    "express the same meaning.\n\n"
    "Garble, fragments, omissions, repetitions, filler words, false starts, and other "
    "disfluencies are not interpretations. They contribute nothing and must never be "
    "reported as alternatives. If one source gives a fluent specific reading while "
    "another is garbled, silent, or merely less complete, there is no disagreement. If "
    "two differently worded readings are equivalent after removing filler and "
    "disfluencies, there is no disagreement.\n\n"
    "For EACH proposed alternative, first remove filler, repetitions, and false starts, "
    "then ask: (1) does this source state a complete intelligible claim, and (2) would "
    "accepting it change what the passage means? If either answer is no, omit it. Never "
    "paste confusing source words merely because they differ from the cleaned words. A "
    "valid alternative's readable text must stand on its own as grammatical English. If "
    "you cannot state its meaning fluently without guessing words absent from the source, "
    "the source is garbled and the alternative does not exist.\n\n"
    "Do NOT report:\n"
    "- Text present in one source and merely absent from the others. Recognizers differ "
    "in sensitivity; a quiet passage only one heard is normally real.\n"
    "- Paraphrases, wording, spelling, or punctuation differences that preserve meaning.\n"
    "- Filler, repetition, false starts, or other disfluencies the cleanup removed.\n"
    "- Garbled or incomplete source text that does not state a coherent competing "
    "meaning.\n\n"
    "Answer with exactly this JSON shape:\n"
    '{"uncertain": [{"quote": "...", "before": "...", "after": "...", '
    '"alternatives": [{"text": "...", "source": 1, "verbatim": "..."}]}]}\n'
    '- "quote": copied character-for-character from the CLEANED TRANSCRIPT; the '
    "shortest span that covers the doubtful claim (a clause, not a paragraph), at most "
    f"{ANNOTATION_MAX_QUOTE_CHARS} characters.\n"
    '- "before" and "after": the few words of the CLEANED TRANSCRIPT immediately '
    "around the quote, copied exactly. They pin WHICH occurrence you mean when the "
    'same words appear more than once; use "" only at the very start or end.\n'
    '- Each alternative cites its source: "verbatim" is the competing text COPIED '
    'EXACTLY from one source at that same moment, "source" is that source\'s number, '
    'and "text" is the shortest fluent, filler-free phrase that states ONLY the '
    'changed meaning, without shared surrounding content. If removing meaning shared '
    'with the quote leaves no distinct claim, omit the alternative. Use "" only when '
    'the verbatim is already that minimal fluent phrase. '
    "An alternative you cannot copy out of a source does not exist — "
    "leave it out. A source that is silent at that moment contributes nothing: give "
    f"fewer alternatives (at most {ANNOTATION_MAX_ALTERNATIVES}) rather than reach "
    "elsewhere in the recording.\n"
    "- Skip an entry unless at least one alternative states a coherent, materially "
    "different meaning from the quote.\n"
    "- Order entries as they appear in the cleaned transcript.\n\n"
    "Calibration: report every semantic divergence, but do not aim for any particular "
    "number of entries. A difficult recording may still yield an empty list when its "
    "differences are only garble, omissions, filler, or equivalent wording.\n\n"
    "Corroboration is semantic, not literal: differently worded sources corroborate one "
    "another when they express the same meaning. When two sources support one meaning "
    "and a third coherently states an incompatible meaning at that same moment, report "
    "the incompatible reading. A majority of fallible recognizers is evidence, not proof."
)

_ASK = "Report only coherent, materially incompatible source meanings."

# The worked example (synthetic on purpose). It demonstrates: several entries from one
# recording, clause-sized quotes, alternatives read off the same moment, a garbled
# source contributing nothing (recognizer-b trails off where "grilled" is disputed), a
# fluent reading corroborated only by omission and garble ("cousin Dana") not reported,
# and agreed text ("for everyone") not reported.
EXAMPLE_USER = (
    "Source 1 — recognizer-a:\n"
    "so we drove out to the lake house on friday and my brother grilled uh grilled fish "
    "for everyone then my cousin dana called from the airport\n\n"
    "Source 2 — recognizer-b:\n"
    "so we drove to the lake house on sunday and my brother uh for everyone\n\n"
    "Source 3 — recognizer-c:\n"
    "so we drove out to the lake house on friday and my brother um brought fish for "
    "everyone then my cousin down at the air something\n\n"
    "CLEANED TRANSCRIPT:\n"
    "We drove out to the lake house on Friday, and my brother grilled fish for everyone. "
    "Then my cousin Dana called from the airport.\n\n"
    f"{_ASK}"
)

EXAMPLE_ASSISTANT = (
    '{"uncertain": ['
    '{"quote": "on Friday", '
    '"before": "out to the lake house", "after": ", and my brother", '
    '"alternatives": [{"text": "on Sunday", "source": 2, '
    '"verbatim": "on sunday and my brother"}]}, '
    '{"quote": "my brother grilled fish", '
    '"before": "on Friday, and", "after": "for everyone.", '
    '"alternatives": [{"text": "my brother brought fish", "source": 3, '
    '"verbatim": "my brother um brought fish"}]}'
    "]}"
)


# The schema is sent as response_format, so the model is *constrained* to it rather than
# merely asked. maxLength/maxItems are what bound the worst-case reply length. Every
# property is in "required" because strict-mode providers demand it; "" plays absent.
def response_schema(max_notes: int = ANNOTATION_MAX_NOTES) -> dict[str, Any]:
    """The reply schema, with the note cap scaled to the transcript being audited
    (an hour of unclear audio legitimately carries more divergences than a memo)."""
    citation = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "maxLength": ANNOTATION_MAX_QUOTE_CHARS,
                "description": (
                    "Shortest standalone grammatical, filler-free phrase stating only "
                    "the meaning that differs from the quote; omit shared semantic "
                    "content."
                ),
            },
            "source": {"type": "integer", "minimum": 0},
            "verbatim": {"type": "string", "maxLength": ANNOTATION_MAX_VERBATIM_CHARS},
        },
        "required": ["text", "source", "verbatim"],
        "additionalProperties": False,
    }
    entry = {
        "type": "object",
        "properties": {
            "quote": {"type": "string", "maxLength": ANNOTATION_MAX_QUOTE_CHARS},
            "before": {"type": "string", "maxLength": ANNOTATION_MAX_CONTEXT_CHARS},
            "after": {"type": "string", "maxLength": ANNOTATION_MAX_CONTEXT_CHARS},
            "alternatives": {
                "type": "array",
                "minItems": 1,
                "maxItems": ANNOTATION_MAX_ALTERNATIVES,
                "description": (
                    "Only coherent source readings whose meanings materially "
                    "contradict the cleaned quote; never garble, filler, repetition, "
                    "or equivalent wording."
                ),
                "items": citation,
            },
        },
        "required": ["quote", "before", "after", "alternatives"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "uncertain": {"type": "array", "maxItems": max_notes, "items": entry}
        },
        "required": ["uncertain"],
        "additionalProperties": False,
    }


def response_format(max_notes: int = ANNOTATION_MAX_NOTES) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "uncertain_passages",
            "strict": True,
            "schema": response_schema(max_notes),
        },
    }


def notes_cap(cleaned_text: str) -> int:
    """The note budget for one transcript: the base cap, plus one per
    ANNOTATION_WORDS_PER_NOTE words, up to the ceiling."""
    return min(
        ANNOTATION_MAX_NOTES_CEILING,
        max(ANNOTATION_MAX_NOTES, len(cleaned_text.split()) // ANNOTATION_WORDS_PER_NOTE),
    )


# The defaults, for callers (and tests) that treat the contract as static.
RESPONSE_SCHEMA: dict[str, Any] = response_schema()
RESPONSE_FORMAT: dict[str, Any] = response_format()


def build_prompt(cleaned_text: str, sources: "list[Transcript]") -> str:
    """The data-only user turn. All rules live in SYSTEM_PROMPT; mirroring them here
    taught nothing and diluted the contract (see the prompt-architecture note above).

    Known per-source pathologies (quality_hint, e.g. "tends to repeat phrases") ride
    along in the header — the cleanup prompt already gets them, and the auditor can
    weigh a source's testimony better knowing how it usually fails."""
    blocks = []
    for index, source in enumerate(sources, start=1):
        header = f"Source {index} — {source.label} ({source.model})"
        if source.quality_hint:
            header += f" — {source.quality_hint}"
        blocks.append(f"{header}:\n{source.text}")
    joined = "\n\n".join(blocks)
    return f"{joined}\n\nCLEANED TRANSCRIPT:\n{cleaned_text}\n\n{_ASK}"


@dataclasses.dataclass(frozen=True)
class Citation:
    """A competing reading, cited: ``verbatim`` is the exact span the model copied out
    of a source, ``text`` the readable form shown to the reader. The citation is what
    makes an alternative admissible at all — verify_notes drops anything whose verbatim
    cannot be found in a source, which keeps invented alternatives out without any
    linguistic judgement in code."""

    text: str
    source: int  # 1-based source number as the model claimed it; informational
    verbatim: str

    def display(self) -> str:
        return self.text or self.verbatim


@dataclasses.dataclass(frozen=True)
class UncertaintyNote:
    quote: str
    alternatives: tuple[Citation, ...]
    # The quote's surroundings in the cleaned transcript, used to pin WHICH occurrence
    # of a repeated phrase is meant. Empty at the transcript edges (or from a model
    # that did not fill them in, in which case placement falls back to reading order).
    before: str = ""
    after: str = ""

    def listed(self, number: int, anchored: bool) -> str:
        # "sources also heard", not "unclear audio": when one recognizer invented text
        # the audio may have been perfectly clear, and the note must not blame the
        # recording for a model's guess.
        joined = " / ".join(f'"{alt.display()}"' for alt in self.alternatives)
        suffix = "" if anchored else " (could not anchor this in the text above)"
        return f'[{number}] "{self.quote}" — sources also heard: {joined}{suffix}'


NOTES_HEADING = "Unclear passages:"


@dataclasses.dataclass
class AnnotationResult:
    """What the pass did. ``text`` is the transcript, annotated or untouched."""

    text: str
    inline_count: int = 0
    appended_count: int = 0
    # Alternatives whose citation could not be found in any source (see verify_notes):
    # dropped, never shown, but counted so the eval and diagnostics can see which
    # models fabricate.
    rejected_citations: int = 0
    # The applied notes as plain data (quote, alternatives, anchored), for diagnostics —
    # post-hoc analysis should never have to parse anchors back out of rendered text.
    notes: list[dict[str, object]] = dataclasses.field(default_factory=list)
    error: str | None = None

    @property
    def total(self) -> int:
        return self.inline_count + self.appended_count


def _comparable(text: str) -> str:
    """Loose form for deciding whether two readings actually differ."""
    return re.sub(r"[^\w ]+", "", " ".join(text.casefold().split()))


def _citation(candidate: object) -> Citation | None:
    """One alternative from the reply. A bare string (schemaless provider, or the old
    shape) is tolerated as an uncited citation — verify_notes decides its fate."""
    if isinstance(candidate, str):
        text = " ".join(candidate.split())
        return Citation(text=text, source=0, verbatim="") if text else None
    if not isinstance(candidate, dict):
        return None
    text = " ".join(str(candidate.get("text") or "").split())
    verbatim = " ".join(str(candidate.get("verbatim") or "").split())
    source = candidate.get("source")
    if not text and not verbatim:
        return None
    return Citation(
        text=text, source=source if isinstance(source, int) else 0, verbatim=verbatim
    )


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
    seen_notes: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        quote = entry.get("quote")
        raw = entry.get("alternatives")
        if not isinstance(quote, str) or not quote.strip():
            continue
        if isinstance(raw, (str, dict)):
            raw = [raw]
        if not isinstance(raw, list):
            continue
        quote = quote.strip()
        seen = {_comparable(quote)}
        alternatives: list[Citation] = []
        for candidate in raw:
            citation = _citation(candidate)
            if citation is None:
                continue
            key = _comparable(citation.display())
            if not key or key in seen:
                continue
            seen.add(key)
            alternatives.append(citation)
            if len(alternatives) == ANNOTATION_MAX_ALTERNATIVES:
                break
        if not alternatives:
            continue
        note = UncertaintyNote(
            quote,
            tuple(alternatives),
            before=" ".join(str(entry.get("before") or "").split()),
            after=" ".join(str(entry.get("after") or "").split()),
        )
        # Degenerate repetition guard, measured on a 9B model: the same note emitted
        # 14 times in one reply. Same quote + same context + same readings = one note;
        # a genuine second instance of a repeated phrase differs in before/after.
        key = (
            _comparable(note.quote),
            _comparable(f"{note.before}|{note.after}"),
            _comparable(" / ".join(alt.display() for alt in note.alternatives)),
        )
        if key in seen_notes:
            continue
        seen_notes.add(key)
        notes.append(note)
        if len(notes) == ANNOTATION_MAX_NOTES_CEILING:
            # The schema bounds enforced providers; this bounds the schemaless ones,
            # which could otherwise anchor an unlimited number of notes.
            break
    return notes


def _word_key(text: str) -> str:
    """Tokenized form for exact word-sequence containment: alphanumeric words only,
    casefolded, space-joined. Mechanical — no stemming, no synonyms, no fuzz."""
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.casefold()))


def verify_notes(
    notes: list[UncertaintyNote], sources: "list[Transcript]"
) -> tuple[list[UncertaintyNote], int]:
    """(kept notes, rejected alternative count). An alternative survives only if its
    citation actually appears in some source as a contiguous word sequence.

    The claimed source number is deliberately not trusted or required to match — a
    citation found in any source is real text either way, and a wrong index should not
    kill a genuine reading. This is what makes invented alternatives impossible without
    putting linguistic judgement in code: the model judges, the code looks the citation
    up. A citation-less alternative (bare string from a schemaless provider) gets the
    same lookup on its display text, so an exact copy still verifies.
    """
    haystacks = [f" {_word_key(source.text)} " for source in sources]

    def in_sources(fragment: str) -> bool:
        needle = _word_key(fragment)
        return bool(needle) and any(f" {needle} " in haystack for haystack in haystacks)

    def verified(alt: Citation) -> Citation | None:
        if in_sources(alt.verbatim):
            # The reader sees ``text``, but only ``verbatim`` was verified — an
            # unfaithful "readable form" would smuggle unvetted words past the
            # citation check. A legitimate cleanup only DELETES fillers, so every
            # displayed word must occur in order, with no extra repetitions.
            # Set containment would allow "Alice called Bob" to become
            # "Bob called Alice" despite reversing the claim.
            remaining = iter(_word_key(alt.verbatim).split())
            if alt.text and not all(
                any(word == candidate for candidate in remaining)
                for word in _word_key(alt.text).split()
            ):
                return dataclasses.replace(alt, text="")
            return alt
        if in_sources(alt.text):
            # The display text is itself source text (the bare-string shape). Any
            # accompanying verbatim failed the lookup and must not survive into
            # diagnostics as though it were a citation.
            return dataclasses.replace(alt, verbatim="")
        return None

    kept: list[UncertaintyNote] = []
    rejected = 0
    for note in notes:
        surviving = tuple(
            checked for alt in note.alternatives if (checked := verified(alt)) is not None
        )
        rejected += len(note.alternatives) - len(surviving)
        if surviving:
            kept.append(dataclasses.replace(note, alternatives=surviving))
    return kept, rejected


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


def _flexible_pattern(fragment: str) -> str:
    """Whitespace-flexible regex for a fragment, exact otherwise."""
    return r"\s+".join(re.escape(word) for word in fragment.split())


def _locate(text: str, note: UncertaintyNote, start: int = 0) -> tuple[int, int] | None:
    """Find a note's quote, most-pinned reading first.

    1. Context match: before + quote + after as the model cited them from the cleaned
       transcript. This is what disambiguates repeated phrases — the words around each
       occurrence differ even when the quote doesn't.
    2. Exact / whitespace-flexible quote match from ``start`` (the end of the previous
       placed note; the prompt requires entries in transcript order), falling back to a
       global search.

    Nothing fuzzier than whitespace and case on purpose: an anchor attached to the
    wrong sentence is worse than an unanchored listing, and an unlocatable quote is
    never lost — it stays in the notes list.
    """
    quote_pattern = _flexible_pattern(note.quote)
    if not quote_pattern:
        return None
    if note.before or note.after:
        parts = []
        if note.before:
            parts.append(_flexible_pattern(note.before) + r"\W{0,4}")
        parts.append(f"({quote_pattern})")
        if note.after:
            parts.append(r"\W{0,4}" + _flexible_pattern(note.after))
        match = re.search(r"\s*".join(parts), text, re.IGNORECASE)
        if match:
            return match.start(1), match.end(1)
    index = text.find(note.quote, start)
    if index == -1:
        index = text.find(note.quote)
    if index != -1:
        return index, index + len(note.quote)
    # Word tokens joined by any non-word run: tolerates punctuation-style differences
    # (a model writing 'single' where the transcript has "double" quotes) while still
    # requiring the exact word sequence. Measured: a real quote failed to anchor over
    # exactly that.
    # Interior apostrophes stay inside a token ("let's"); a bare ' is a quotation
    # mark, not a word, and must not become a token of its own.
    tokens = re.findall(r"\w+(?:'\w+)*", note.quote)
    word_pattern = r"\W+".join(re.escape(token) for token in tokens) if tokens else quote_pattern
    for pattern in (quote_pattern, word_pattern):
        match = re.search(pattern, text[start:], re.IGNORECASE)
        if match:
            return start + match.start(), start + match.end()
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start(), match.end()
    return None


def _merge(base: UncertaintyNote, extra: UncertaintyNote) -> UncertaintyNote:
    """Fold an overlapping note's alternatives into the one already anchored there."""
    seen = {_comparable(base.quote)} | {_comparable(alt.display()) for alt in base.alternatives}
    merged = list(base.alternatives)
    for alt in extra.alternatives:
        key = _comparable(alt.display())
        if key and key not in seen:
            seen.add(key)
            merged.append(alt)
    return dataclasses.replace(base, alternatives=tuple(merged))


def apply_notes(text: str, notes: list[UncertaintyNote]) -> AnnotationResult:
    """Anchor each located quote with [n]; list every note, anchored or not, at the end.

    The transcript body is never altered — anchors are only inserted — so this cannot
    lose content even if the auditor misbehaves. Numbers follow body order; notes that
    could not be anchored take the numbers after the last anchored one. Notes whose
    spans overlap are merged into one anchor rather than one of them losing its place.
    """
    located: list[tuple[int, int, UncertaintyNote]] = []
    unplaced: list[UncertaintyNote] = []
    cursor = 0
    for note in notes:
        span = _locate(text, note, cursor)
        if span is None:
            unplaced.append(note)
            continue
        start, end = span
        cursor = max(cursor, end)
        overlap = next(
            (
                index
                for index, (other_start, other_end, _) in enumerate(located)
                if start < other_end and other_start < end
            ),
            None,
        )
        if overlap is not None:
            other_start, other_end, other = located[overlap]
            located[overlap] = (other_start, other_end, _merge(other, note))
            continue
        located.append((start, end, note))

    located.sort(key=lambda item: item[0])
    annotated = text
    for number, (_start, end, _note) in reversed(list(enumerate(located, start=1))):
        annotated = f"{annotated[:end]}[{number}]{annotated[end:]}"

    lines = [note.listed(number, True) for number, (_, _, note) in enumerate(located, start=1)]
    lines += [
        note.listed(number, False)
        for number, note in enumerate(unplaced, start=len(located) + 1)
    ]
    if lines:
        annotated = f"{annotated.rstrip()}\n\n{NOTES_HEADING}\n" + "\n".join(lines)

    def payload(note: UncertaintyNote, anchored: bool) -> dict[str, object]:
        return {
            "quote": note.quote,
            "anchored": anchored,
            "alternatives": [
                {"text": alt.text, "source": alt.source, "verbatim": alt.verbatim}
                for alt in note.alternatives
            ],
        }

    return AnnotationResult(
        text=annotated,
        inline_count=len(located),
        appended_count=len(unplaced),
        notes=[payload(note, True) for _, _, note in located]
        + [payload(note, False) for note in unplaced],
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
    cap = notes_cap(cleaned_text)
    # The output allowance grows with the note cap: a bounded schema is no protection
    # if a full legitimate answer cannot fit in the tokens requested for it.
    wanted = max(ANNOTATION_MAX_OUTPUT_TOKENS, cap * ANNOTATION_TOKENS_PER_NOTE)
    requested = min(wanted, max(1024, budget - estimated))
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
            response_format=response_format(cap),
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
    notes, rejected = verify_notes(notes, sources)
    if not notes:
        return AnnotationResult(text=cleaned_text, rejected_citations=rejected)
    result = apply_notes(cleaned_text, notes)
    result.rejected_citations = rejected
    return result
