"""Hybrid retrieval: lexical + vector channels, deduplicated to concepts.

Per description:   lexical (0..1), semantic (cosine 0..1), exact bonus
Per concept:       best of its descriptions across all query variants
retrieval_score =  w_lex*lexical + w_sem*semantic + w_exact*exact + w_prior*prior
                   (weights renormalised when no vector index is configured)
                   x soft semantic-category factor

The pool is deliberately large (~200 descriptions per channel and variant)
before deduplication to ~32 concepts: this stage optimises recall.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from clif.config import Settings
from clif.retrieval.category import CategoryPrediction, category_of_tag
from clif.retrieval.lexical import DescriptionHit, LexicalRetriever
from clif.retrieval.vector import CachedEmbedder, VectorIndex
from clif.terminology.hierarchy import HierarchyConstraint
from clif.terminology.store import TerminologyStore
from clif.text import QueryText


@dataclass
class MatchedDescription:
    description_id: str
    term: str
    language: str
    type: str
    is_pt: bool
    channels: list[str]
    lexical: float
    semantic: float
    exact: float
    variant: str


@dataclass
class RetrievedConcept:
    concept_id: str
    fsn: str
    preferred_term: str
    semantic_tag: str | None
    category: str
    score: float
    lexical: float
    semantic: float
    exact: float
    prior: float
    category_factor: float
    matched: list[MatchedDescription]
    rank: int = 0

    @property
    def best(self) -> MatchedDescription:
        return self.matched[0]


@dataclass
class RetrievalResult:
    candidates: list[RetrievedConcept]
    descriptions_considered: int
    concepts_considered: int
    filtered: list[dict] = field(default_factory=list)
    languages: list[str] | None = None
    channel_counts: dict[str, int] = field(default_factory=dict)
    vector_enabled: bool = False
    elapsed_ms: float = 0.0
    cached: bool = False


class _LRU:
    def __init__(self, size: int):
        self.size = size
        self.data: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            v = self.data.get(key)
            if v is not None:
                self.data.move_to_end(key)
            return v

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)


class HybridRetriever:
    def __init__(self, store: TerminologyStore, settings: Settings, vector_index: VectorIndex | None = None,
                 embedder: CachedEmbedder | None = None):
        self.store = store
        self.settings = settings
        self.lexical = LexicalRetriever(store, settings.description_pool, settings.bm25_common_df_ratio,
                                        settings.fuzzy_min_similarity, settings.fuzzy_min_ratio)
        self.vector_index = vector_index
        self.embedder = embedder
        self._cache = _LRU(settings.retrieval_cache_size)

    @property
    def vector_enabled(self) -> bool:
        return self.vector_index is not None and self.embedder is not None

    def retrieve(
        self,
        variants: list[QueryText],
        languages: list[str] | None,
        limit: int | None = None,
        constraint: HierarchyConstraint | None = None,
        category: CategoryPrediction | None = None,
        exclude_tags: frozenset[str] = frozenset(),
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        limit = limit or self.settings.candidate_count
        key = (
            tuple((v.norm, round(v.weight, 3)) for v in variants),
            tuple(languages) if languages else None,
            limit,
            (tuple(constraint.allowed_roots), constraint.ecl_text) if constraint else None,
            tuple(sorted((c, round(p, 2)) for c, p in category.probs.items())) if category else None,
            exclude_tags,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return RetrievalResult(**{**cached.__dict__, "cached": True, "elapsed_ms": 0.0})

        pool = self.settings.description_pool * (3 if constraint else 1)
        hits: dict[int, DescriptionHit] = {}
        channel_counts: dict[str, int] = {}

        def merge(h: DescriptionHit, weight: float) -> None:
            cur = hits.get(h.description.rowid)
            scaled_lex, scaled_sem, scaled_exact = h.lexical * weight, h.semantic * weight, h.exact * weight
            if cur is None:
                h.lexical, h.semantic, h.exact = scaled_lex, scaled_sem, scaled_exact
                hits[h.description.rowid] = h
                return
            cur.channels |= h.channels
            if scaled_lex > cur.lexical:
                cur.lexical, cur.variant = scaled_lex, h.variant
            cur.semantic = max(cur.semantic, scaled_sem)
            cur.exact = max(cur.exact, scaled_exact)

        for v in variants:
            for h in self.lexical.search(v, languages, pool).values():
                for ch in h.channels:
                    channel_counts[ch] = channel_counts.get(ch, 0) + 1
                merge(h, v.weight)

        if self.vector_enabled:
            vecs = self.embedder.embed_many([v.text for v in variants])  # type: ignore[union-attr]
            vec_hits: dict[int, tuple[float, str, float]] = {}
            for v, qv in zip(variants, vecs):
                for rowid, cos in self.vector_index.search(qv, pool):  # type: ignore[union-attr]
                    cos = max(0.0, min(1.0, cos))
                    if rowid not in vec_hits or cos * v.weight > vec_hits[rowid][0] * vec_hits[rowid][2]:
                        vec_hits[rowid] = (cos, v.text, v.weight)
            missing = [r for r in vec_hits if r not in hits]
            descs = self.store.descriptions_by_rowids(missing)
            for rowid, (cos, vtext, w) in vec_hits.items():
                if rowid in hits:
                    hits[rowid].semantic = max(hits[rowid].semantic, cos * w)
                    hits[rowid].channels.add("vector")
                elif rowid in descs:
                    d = descs[rowid]
                    if languages and d.language not in languages:
                        continue
                    hits[rowid] = DescriptionHit(d, {"vector"}, semantic=cos * w, variant=vtext)
                channel_counts["vector"] = channel_counts.get("vector", 0) + 1

        # --- aggregate by concept -------------------------------------------
        by_concept: dict[str, list[DescriptionHit]] = {}
        for h in hits.values():
            by_concept.setdefault(h.description.concept_id, []).append(h)

        w = self.settings.weights
        w_sem = w.semantic if self.vector_enabled else 0.0
        w_total = w.lexical + w_sem + w.exact + w.prior
        results: list[RetrievedConcept] = []
        filtered: list[dict] = []
        hard_cat = (
            category.top
            if category and self.settings.category_hard_filter_enabled
            and category.certainty >= self.settings.category_hard_filter_certainty
            else None
        )
        lang_rank = {lang: i for i, lang in enumerate(languages or [self.settings.default_display_language])}
        for cid, dhits in by_concept.items():
            rec = self.store.concept(cid)
            if rec is None or not rec.active or not rec.fsn:
                filtered.append({"concept_id": cid, "reason": "inactive_or_unknown"})
                continue
            if rec.semantic_tag in exclude_tags:
                filtered.append({"concept_id": cid, "reason": f"excluded_tag:{rec.semantic_tag}"})
                continue
            if constraint is not None:
                ok, why = constraint.allows(cid, self.store)
                if not ok:
                    filtered.append({"concept_id": cid, "reason": why})
                    continue
            cat = category_of_tag(rec.semantic_tag)
            if hard_cat and cat != hard_cat and category.support(cat) < 0.05:  # type: ignore[union-attr]
                filtered.append({"concept_id": cid, "reason": f"category_hard_filter:{hard_cat}"})
                continue
            dhits.sort(key=lambda h: (round(h.lexical + h.semantic + h.exact, 6), h.description.is_pt,
                                      -lang_rank.get(h.description.language, 99)), reverse=True)
            lex = max(h.lexical for h in dhits)
            sem = max(h.semantic for h in dhits)
            exact = max(h.exact for h in dhits)
            prior = self.settings.semantic_tag_priors.get(rec.semantic_tag or "", 0.5)
            if any(h.description.is_pt and h.exact >= 0.9 for h in dhits):
                prior = min(1.0, prior + 0.1)
            raw = (w.lexical * lex + w_sem * sem + w.exact * exact + w.prior * prior) / w_total
            factor = 1.0
            if category is not None and self.settings.category_weight > 0:
                top_support = category.support(category.top) or 1.0
                ratio = min(1.0, category.support(cat) / top_support)
                factor = (1 - self.settings.category_weight) + self.settings.category_weight * ratio
            display_lang = self.settings.default_display_language
            pt, _ = rec.preferred_term(display_lang)
            results.append(
                RetrievedConcept(
                    concept_id=cid, fsn=rec.fsn, preferred_term=pt, semantic_tag=rec.semantic_tag, category=cat,
                    score=raw * factor, lexical=lex, semantic=sem, exact=exact, prior=prior, category_factor=factor,
                    matched=[
                        MatchedDescription(h.description.description_id, h.description.term, h.description.language,
                                           h.description.type, h.description.is_pt, sorted(h.channels),
                                           round(h.lexical, 4), round(h.semantic, 4), round(h.exact, 4), h.variant)
                        for h in dhits[:5]
                    ],
                )
            )
        results.sort(key=lambda r: (-r.score, r.concept_id))
        results = results[:limit]
        for i, r in enumerate(results, 1):
            r.rank = i
        out = RetrievalResult(
            candidates=results, descriptions_considered=len(hits), concepts_considered=len(by_concept),
            filtered=filtered[:200], languages=languages, channel_counts=channel_counts,
            vector_enabled=self.vector_enabled, elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
        self._cache.put(key, out)
        return out
