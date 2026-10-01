"""Constrained-choice reranking interface.

A reranker never searches SNOMED. It receives the phrase, optional context and
at most ~32 candidates, and returns a score for every option: each candidate
plus NO_MATCH and AMBIGUOUS. Options are addressed by opaque labels (A, B, ...)
and mapped back to SCTIDs here, so a model cannot emit an SCTID that was not
supplied. Unknown labels are dropped and reported.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from string import ascii_uppercase

NO_MATCH = "NO_MATCH"
AMBIGUOUS = "AMBIGUOUS"

SPECIFICITY_INSTRUCTIONS = (
    "Select the single SNOMED CT concept that best represents the clinical text. "
    "Choose the MOST SPECIFIC concept that is fully supported by the text, but never a concept that adds "
    "details absent from the text or context (laterality, severity, acuity, anatomical site, cause, stage, "
    "recurrence, complications). Example: 'appendicitis' -> 'Appendicitis', not 'Acute appendicitis'. "
    "Negation, uncertainty and history have already been separated from the text; map the underlying concept. "
    "Answer NO_MATCH if none of the options represents the text. Answer AMBIGUOUS if the text does not "
    "provide enough information to choose between options reliably (e.g. an abbreviation with several "
    "plausible meanings)."
)


@dataclass
class Candidate:
    label: str
    concept_id: str
    preferred_term: str
    fsn: str
    semantic_tag: str | None
    synonyms: list[str] = field(default_factory=list)
    hierarchy: list[str] = field(default_factory=list)  # parent preferred terms
    matched_terms: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0
    retrieval_rank: int = 0


@dataclass
class RerankRequest:
    text: str  # original input, untouched
    core_text: str  # concept-bearing part after assertion cues were removed
    context: str | None
    candidates: list[Candidate]
    expanded_forms: list[str] = field(default_factory=list)
    language: str | None = None

    def by_label(self) -> dict[str, Candidate]:
        return {c.label: c for c in self.candidates}


@dataclass
class RerankResult:
    scores: dict[str, float]  # concept_id -> probability
    no_match: float
    ambiguous: float
    backend: str
    model: str | None = None
    scores_are_probabilities: bool = True
    explanation: str | None = None
    rejected: list[str] = field(default_factory=list)  # labels/ids returned but not supplied
    latency_ms: float = 0.0
    fallback_used: bool = False
    raw_label_scores: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)  # backend-specific details for debug output
    warnings: list[str] = field(default_factory=list)  # e.g. "option_collapse" forces AMBIGUOUS

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: (-kv[1], kv[0]))

    @property
    def chosen(self) -> str:
        best_c = self.ranked()[0] if self.scores else (None, -1.0)
        options = [(best_c[0], best_c[1]), (NO_MATCH, self.no_match), (AMBIGUOUS, self.ambiguous)]
        return max(options, key=lambda x: x[1])[0] or NO_MATCH


class ConceptReranker(ABC):
    """Swappable reranking backend: rerank(request) -> RerankResult."""

    name: str = "base"
    model: str | None = None

    @abstractmethod
    def rerank(self, request: RerankRequest) -> RerankResult: ...

    def close(self) -> None:  # pragma: no cover - optional
        pass


# Kept for readers of the original design brief.
SnomedChoiceReranker = ConceptReranker


class RerankerError(RuntimeError):
    pass


def option_labels(n: int) -> list[str]:
    labels = list(ascii_uppercase)
    for a in ascii_uppercase:
        for b in ascii_uppercase:
            labels.append(a + b)
    if n > len(labels):
        raise ValueError("too many candidates")
    return labels[:n]


def build_choice_question(request: RerankRequest, include_hierarchy: bool = True) -> dict:
    """Generic constrained-choice payload (used by the LLM adapter)."""
    options: dict[str, dict] = {}
    for c in request.candidates:
        opt = {
            "concept_id": c.concept_id,
            "term": c.preferred_term,
            "fsn": c.fsn,
            "semantic_tag": c.semantic_tag,
        }
        if c.synonyms:
            opt["synonyms"] = c.synonyms[:5]
        if include_hierarchy and c.hierarchy:
            opt["parents"] = c.hierarchy[:3]
        options[c.label] = opt
    options[NO_MATCH] = {"description": "None of the supplied SNOMED concepts accurately represent the text"}
    options[AMBIGUOUS] = {"description": "The text does not provide enough information to select one concept reliably"}
    state = {"text": request.core_text, "original_text": request.text, "context": request.context or ""}
    if request.expanded_forms:
        state["abbreviation_expansions"] = request.expanded_forms
    return {
        "state": state,
        "question": {
            "type": "choice",
            "name": "snomed_concept",
            "instructions": SPECIFICITY_INSTRUCTIONS,
            "options": options,
        },
    }


def to_probabilities(label_scores: dict[str, float]) -> tuple[dict[str, float], bool]:
    """Accept probabilities, unnormalised scores or logits; return a distribution."""
    if not label_scores:
        return {}, False
    vals = list(label_scores.values())
    if any(v < 0 for v in vals) or any(v > 1.0 + 1e-6 for v in vals):
        m = max(vals)
        exps = {k: math.exp(v - m) for k, v in label_scores.items()}
        z = sum(exps.values())
        return {k: v / z for k, v in exps.items()}, True
    total = sum(vals)
    if total <= 0:
        return {k: 1.0 / len(vals) for k in label_scores}, False
    return {k: v / total for k, v in label_scores.items()}, True


def result_from_label_scores(
    request: RerankRequest,
    label_scores: dict[str, float],
    backend: str,
    model: str | None = None,
    explanation: str | None = None,
    scores_are_probabilities: bool = True,
    latency_ms: float = 0.0,
) -> RerankResult:
    """Map label scores back to SCTIDs, rejecting anything that was not offered."""
    by_label = request.by_label()
    by_id = {c.concept_id: c for c in request.candidates}
    clean: dict[str, float] = {}
    rejected: list[str] = []
    for label, score in label_scores.items():
        key = str(label).strip()
        if key in by_label or key in (NO_MATCH, AMBIGUOUS):
            clean[key] = float(score)
        elif key in by_id:  # backend answered with an SCTID that was offered
            clean[by_id[key].label] = float(score)
        else:
            rejected.append(key)
    probs, ok = to_probabilities(clean)
    scores = {by_label[k].concept_id: v for k, v in probs.items() if k in by_label}
    for c in request.candidates:
        scores.setdefault(c.concept_id, 0.0)
    return RerankResult(
        scores=scores,
        no_match=probs.get(NO_MATCH, 0.0),
        ambiguous=probs.get(AMBIGUOUS, 0.0),
        backend=backend,
        model=model,
        scores_are_probabilities=scores_are_probabilities and ok,
        explanation=explanation,
        rejected=rejected,
        latency_ms=latency_ms,
        raw_label_scores=clean,
    )
