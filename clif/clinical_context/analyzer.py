"""Separates the concept-bearing core of a phrase from its assertion context.

"patient denies chest pain"         -> core "chest pain", NEGATED
"history of myocardial infarction"  -> core "myocardial infarction", HISTORICAL
"V.a. Pneumonie"                    -> core "Pneumonie", POSSIBLE
"no known drug allergies"           -> fixed expression, mapped as a whole

The original text is never modified; the analysis records every cue span so
the decision is auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from clif.clinical_context.abbreviation import AbbreviationHit, AbbreviationResolver
from clif.clinical_context.assertion import CUE_EFFECTS as ASSERTION_EFFECTS
from clif.clinical_context.assertion import Assertion, Certainty, Experiencer
from clif.clinical_context.temporality import CUE_EFFECTS as TEMPORAL_EFFECTS
from clif.clinical_context.temporality import Temporality
from clif.text import STOPWORDS, QueryText, content_tokens, nfkc, norm_fold, normalize

_EDGE_JUNK = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_LEADING_FUNCTION = {"of", "for", "with", "the", "a", "an", "von", "auf", "mit", "für", "fur", "to", "van", "op", "de",
                     "has", "have", "had", "hat", "is", "was", "presents"}
_SUBJECT_RE = re.compile(r"\s*(?:the\s+)?(?:patient|patientin|pt\.?|pat\.?)\s+", re.I)


@dataclass
class CueSpan:
    category: str
    cue: str
    start: int
    end: int


@dataclass
class ContextAnalysis:
    original: str
    core: str
    assertion: Assertion = Assertion.ASSERTED
    temporality: Temporality = Temporality.CURRENT
    certainty: Certainty = Certainty.CERTAIN
    experiencer: Experiencer = Experiencer.PATIENT
    fixed_expression: bool = False
    cues: list[CueSpan] = field(default_factory=list)
    abbreviations: list[AbbreviationHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_context_cues(self) -> bool:
        return any(c.category not in ("filler",) for c in self.cues)

    @property
    def unresolved_abbreviations(self) -> list[AbbreviationHit]:
        return [a for a in self.abbreviations if a.resolved is None and len(a.plausible) > 1]

    @property
    def insufficient(self) -> bool:
        return not content_tokens(norm_fold(self.core)) or len(norm_fold(self.core).replace(" ", "")) < 2


def _compile(cues: list[str]) -> re.Pattern | None:
    cues = sorted({c.strip() for c in cues if c and c.strip()}, key=len, reverse=True)
    if not cues:
        return None
    alts = [r"\s*".join(re.escape(p) for p in c.split()) for c in cues]
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.I)


class ContextAnalyzer:
    def __init__(self, lexicon: dict, abbreviations: AbbreviationResolver, situation_mode: str = "fixed_expressions"):
        self.situation_mode = situation_mode
        self.abbreviations = abbreviations
        self.fixed = {normalize(x) for x in lexicon.get("fixed_expressions", [])}
        cats: dict[str, list[str]] = {}
        neg = lexicon.get("negation", {})
        cats["negation"] = neg.get("pre", []) + neg.get("post", [])
        for sub, spec in (lexicon.get("temporality") or {}).items():
            cats[f"temporality.{sub}"] = spec.get("pre", []) + spec.get("post", [])
        for sub, spec in (lexicon.get("certainty") or {}).items():
            cats[f"certainty.{sub}"] = spec.get("pre", []) + spec.get("post", [])
        fam = lexicon.get("experiencer_family") or {}
        cats["experiencer_family"] = fam.get("pre", []) + fam.get("post", [])
        self._patterns = {k: p for k, v in cats.items() if (p := _compile(v))}
        self._fillers = _compile(lexicon.get("fillers", []))
        self._terminators = _compile(lexicon.get("terminators", []))

    @classmethod
    def default(cls, situation_mode: str = "fixed_expressions", dominance: float = 0.9,
                cues_path: Path | None = None, abbreviations_path: Path | None = None) -> "ContextAnalyzer":
        if cues_path is None:
            lex = yaml.safe_load(resources.files("clif.resources").joinpath("context_cues.yaml").read_text("utf-8"))
        else:
            lex = yaml.safe_load(Path(cues_path).read_text("utf-8"))
        return cls(lex, AbbreviationResolver.from_yaml(abbreviations_path, dominance), situation_mode)

    # ------------------------------------------------------------------
    def analyze(self, text: str, context: str | None = None) -> ContextAnalysis:
        original = text
        t = nfkc(text).strip()
        result = ContextAnalysis(original=original, core=t)

        if self.situation_mode != "off" and self._is_fixed(t):
            result.fixed_expression = True
            result.abbreviations = self.abbreviations.find(t, context)
            return result

        spans: list[CueSpan] = []
        for cat, pat in self._patterns.items():
            spans.extend(CueSpan(cat, m.group(0), m.start(), m.end()) for m in pat.finditer(t))
        # "?" marks uncertainty ("pneumonia?").
        for m in re.finditer(r"\?", t):
            spans.append(CueSpan("certainty.possible", "?", m.start(), m.end()))
        spans = _non_overlapping(spans)

        # Leading fillers ("patient has", "complains of"), possibly after a cue.
        filler_spans = self._leading_fillers(t, spans)
        all_spans = sorted(spans + filler_spans, key=lambda s: s.start)

        # Clauses separated by terminators ("no fever but cough").
        clauses = self._clauses(t)
        content_clauses = []
        for cs, ce in clauses:
            residue = _remove(t[cs:ce], [(s.start - cs, s.end - cs) for s in all_spans if cs <= s.start < ce])
            if content_tokens(norm_fold(residue)):
                content_clauses.append((cs, ce, residue))
        if len(content_clauses) > 1:
            result.warnings.append("multiple_clauses: only the first clause is mapped")
        if content_clauses:
            cs, ce, residue = content_clauses[0]
            active = [s for s in spans if cs <= s.start < ce]
        else:
            residue, active = "", spans
        result.core = _clean(residue)
        result.cues = [s for s in all_spans if s in active or s in filler_spans]

        effects = {**ASSERTION_EFFECTS, **TEMPORAL_EFFECTS}
        for s in active:
            if s.category in effects:
                attr, value = effects[s.category]
                if attr == "certainty" and result.certainty == Certainty.POSSIBLE:
                    continue  # POSSIBLE wins over PROBABLE
                setattr(result, attr, value)
        if sum(1 for s in active if s.category == "negation") > 1:
            result.warnings.append("multiple_negation_cues")

        result.abbreviations = self.abbreviations.find(result.core, context)
        return result

    def _is_fixed(self, t: str) -> bool:
        n = normalize(_SUBJECT_RE.sub("", t, count=1) if _SUBJECT_RE.match(t) else t)
        return n in self.fixed

    def _leading_fillers(self, t: str, spans: list[CueSpan]) -> list[CueSpan]:
        """Fillers at the start of the phrase, possibly interleaved with cues."""
        out: list[CueSpan] = []
        starts = {s.start: s for s in spans}
        pos = 0
        while True:
            while pos < len(t) and t[pos].isspace():
                pos += 1
            if pos in starts:
                pos = starts[pos].end
                continue
            fm = self._fillers.match(t, pos) if self._fillers else None
            if fm is None:
                fm = _SUBJECT_RE.match(t, pos)
            if fm is None or fm.end() == pos:
                return out
            out.append(CueSpan("filler", fm.group(0).strip(), fm.start(), fm.end()))
            pos = fm.end()

    def _clauses(self, t: str) -> list[tuple[int, int]]:
        if self._terminators is None:
            return [(0, len(t))]
        bounds = [0]
        for m in self._terminators.finditer(t):
            bounds.extend([m.start(), m.end()])
        bounds.append(len(t))
        return [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2)]

    # ------------------------------------------------------------------
    def query_variants(self, a: ContextAnalysis, include_full_text: bool = False) -> list[QueryText]:
        """Retrieval queries: the core, abbreviation expansions, optionally the full text."""
        variants: list[QueryText] = []
        base = a.original if a.fixed_expression else a.core
        if a.abbreviations:
            # Replace abbreviations right-to-left so earlier offsets stay valid.
            texts = [(base, 1.0)]
            for hit in sorted(a.abbreviations, key=lambda h: -h.start):
                choices = [(hit.resolved, 1.0)] if hit.resolved else hit.plausible
                texts = [(tx[: hit.start] + exp + tx[hit.end:], w * p) for tx, w in texts for exp, p in choices]
            for tx, w in texts:
                variants.append(QueryText.build(tx, "abbreviation", max(w, 0.2)))
            # The abbreviation itself is searched too: SNOMED contains synonyms such as "COPD".
            variants.append(QueryText.build(base, "input", 0.6))
        else:
            variants.append(QueryText.build(base, "fixed_expression" if a.fixed_expression else "core", 1.0))
        if include_full_text and not a.fixed_expression and normalize(a.original) != normalize(base):
            variants.append(QueryText.build(a.original, "full_text", 0.8))
        seen: set[str] = set()
        out = []
        for v in variants:
            if v.norm and v.norm not in seen:
                seen.add(v.norm)
                out.append(v)
        return out


def _non_overlapping(spans: list[CueSpan]) -> list[CueSpan]:
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[CueSpan] = []
    last_end = -1
    for s in spans:
        if s.start >= last_end:
            out.append(s)
            last_end = s.end
    return out


def _remove(text: str, ranges: list[tuple[int, int]]) -> str:
    out, pos = [], 0
    for s, e in sorted(ranges):
        if s < pos:
            continue
        out.append(text[pos:s])
        out.append(" ")
        pos = e
    out.append(text[pos:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _clean(text: str) -> str:
    t = _EDGE_JUNK.sub("", text.strip())
    words = t.split()
    while words and words[0].lower() in _LEADING_FUNCTION:
        words = words[1:]
    while words and words[-1].lower() in _LEADING_FUNCTION | STOPWORDS:
        words = words[:-1]
    return _EDGE_JUNK.sub("", " ".join(words))
