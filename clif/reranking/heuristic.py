"""Deterministic reranker (tests, offline development, fallback).

Scores each candidate by bidirectional soft token coverage between the phrase
and the candidate's descriptions:
    recall    = share of phrase tokens the description covers
    precision = share of description tokens supported by the phrase
Low precision means the candidate adds unsupported detail (e.g. "acute"), which
implements the specificity rule without any model. It is a transparent
baseline, not a substitute for a trained constrained-choice model.
"""

from __future__ import annotations

import math
import time

from clif.reranking.base import (
    AMBIGUOUS,
    NO_MATCH,
    Candidate,
    ConceptReranker,
    RerankRequest,
    RerankResult,
    result_from_label_scores,
)
from clif.text import content_tokens, coverage, norm_fold, strip_semantic_tag, tokens


class HeuristicReranker(ConceptReranker):
    name = "heuristic"
    model = "bidirectional-coverage-v1"

    # Sharp temperature: real SNOMED has many near-duplicates (17 kinds of appendicitis), and a
    # soft distribution would spread the exact match's mass across all of them.
    def __init__(self, temperature: float = 30.0, no_match_level: float = 0.55, tie_margin: float = 0.03):
        self.temperature = temperature
        self.no_match_level = no_match_level
        self.tie_margin = tie_margin

    @staticmethod
    def _toks(text: str) -> tuple[str, list[str]]:
        f = norm_fold(text)
        return f, content_tokens(f) or tokens(f)

    def _score(self, queries: list[tuple[str, list[str]]], cand: Candidate, context_toks: list[str]) -> float:
        terms = [cand.preferred_term, strip_semantic_tag(cand.fsn)[0], *cand.synonyms, *cand.matched_terms]
        best = 0.0
        for term in dict.fromkeys(t for t in terms if t):
            df, dt = self._toks(term)
            for qf, qt in queries:
                if df == qf:
                    s = 1.0
                else:
                    s = 0.6 * coverage(qt, dt) + 0.4 * coverage(dt, qt)
                best = max(best, s)
        if best >= 0.5 and context_toks:
            _, pt = self._toks(cand.preferred_term)
            best += 0.03 * coverage(pt, context_toks)
        return best + 0.02 * cand.retrieval_score

    def rerank(self, request: RerankRequest) -> RerankResult:
        t0 = time.perf_counter()
        queries = [self._toks(q) for q in dict.fromkeys([request.core_text, *request.expanded_forms]) if q]
        context_toks = self._toks(request.context)[1] if request.context else []
        raw = {c.label: self._score(queries, c, context_toks) for c in request.candidates}
        ranked = sorted(raw.values(), reverse=True)
        s1 = ranked[0] if ranked else 0.0
        s2 = ranked[1] if len(ranked) > 1 else 0.0
        logits = {label: self.temperature * s for label, s in raw.items()}
        logits[NO_MATCH] = self.temperature * self.no_match_level
        tie = s2 >= self.no_match_level and (s1 - s2) < self.tie_margin
        logits[AMBIGUOUS] = self.temperature * s1 + 0.1 if tie else -math.inf
        m = max(logits.values())
        probs = {k: (math.exp(v - m) if v != -math.inf else 0.0) for k, v in logits.items()}
        z = sum(probs.values())
        probs = {k: v / z for k, v in probs.items()}
        explanation = "tie between top candidates" if tie else None
        return result_from_label_scores(
            request, probs, self.name, self.model, explanation=explanation,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
