"""Lexical retrieval channels over SNOMED descriptions.

A. exact match (normalized form)       -> index lookup
B. accent-folded exact match           -> index lookup
C. prefix match                        -> index range scan
D. fuzzy match                         -> query tokens missing from the term vocabulary are
                                          spelling-corrected against it (rapidfuzz), then the
                                          corrected query re-runs A-C and E
E. BM25 full text                      -> FTS5, planned: a selective AND query first; an OR
                                          query over discriminative tokens only when AND is thin

Why not trigram FTS for fuzziness: on 1.4M descriptions a trigram OR-query touches
hundreds of thousands of rows (~200 ms); correcting tokens against a ~150k-entry
vocabulary costs a few ms and gives the same recall for typos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from clif.terminology.store import Description, TerminologyStore
from clif.text import QueryText, content_tokens, coverage, tokens


@dataclass
class DescriptionHit:
    description: Description
    channels: set[str] = field(default_factory=set)
    exact: float = 0.0  # 1.0 normalized exact, 0.9 folded exact, 0.3 prefix
    bm25: float = 0.0  # raw (positive, higher = better)
    lexical: float = 0.0  # final 0..1
    semantic: float = 0.0  # cosine, 0..1
    variant: str = ""


def _fts_quote(tok: str) -> str:
    return '"' + tok.replace('"', "") + '"'


class LexicalRetriever:
    def __init__(self, store: TerminologyStore, pool: int = 200, common_df_ratio: float = 0.02,
                 fuzzy_min_similarity: float = 80.0, fuzzy_min_ratio: float = 60.0):
        self.store = store
        self.pool = pool
        self.common_df = max(50, int(store.total_terms * common_df_ratio))
        self.fuzzy_min_similarity = fuzzy_min_similarity
        self.fuzzy_min_ratio = fuzzy_min_ratio

    # ------------------------------------------------------------------
    def search(self, q: QueryText, languages: list[str] | None, pool: int | None = None) -> dict[int, DescriptionHit]:
        pool = pool or self.pool
        hits: dict[int, DescriptionHit] = {}
        if not q.folded:
            return hits
        self._run(q, languages, pool, hits, channel_suffix="")
        corrected = self.correct(q)
        if corrected is not None:
            self._run(corrected, languages, pool, hits, channel_suffix="fuzzy")
        self._score(q, hits)
        # Drop weak fuzzy-only hits: they add noise, not recall.
        for rid in [r for r, h in hits.items() if h.channels <= {"fuzzy"} and h.lexical * 100 < self.fuzzy_min_ratio]:
            del hits[rid]
        return hits

    def _run(self, q: QueryText, languages, pool: int, hits: dict[int, DescriptionHit], channel_suffix: str) -> None:
        def hit(d: Description, channel: str) -> DescriptionHit:
            h = hits.get(d.rowid)
            if h is None:
                h = hits[d.rowid] = DescriptionHit(d, variant=q.text)
            h.channels.add(channel_suffix or channel)
            return h

        # Corrected queries earn slightly less than genuine exact matches.
        exact_scale = 0.95 if channel_suffix else 1.0
        for d in self.store.exact(q.norm, q.folded, languages):
            h = hit(d, "exact")
            h.exact = max(h.exact, exact_scale * (1.0 if d.norm == q.norm else 0.9))
        if len(q.folded) >= 4:
            for d in self.store.prefix(q.folded, languages, limit=50):
                h = hit(d, "prefix")
                h.exact = max(h.exact, 0.3)
        for d, score in self._bm25(q, languages, pool):
            h = hit(d, "bm25")
            h.bm25 = max(h.bm25, score)

    def _bm25(self, q: QueryText, languages, pool: int) -> list[tuple[Description, float]]:
        toks = list(dict.fromkeys(q.content or tokens(q.folded)))
        if not toks:
            return []
        df = self.store.vocab_df
        quoted = [_fts_quote(t) for t in toks[:-1]]
        last = toks[-1]
        # Prefix on the last token supports typing-as-you-go and German compounds.
        quoted.append(_fts_quote(last) + ("*" if len(last) >= 3 else ""))
        results = self.store.bm25(" AND ".join(quoted), languages, pool) if len(toks) > 1 else []
        if len(results) >= pool // 4:
            return results
        # OR over discriminative tokens. Very common tokens rank nothing but touch huge posting lists.
        selective = [t for t in toks if df.get(t, 0) <= self.common_df]
        if not selective:  # all tokens common: keep the rarest one
            selective = [min(toks, key=lambda t: df.get(t, 0))]
        if len(toks) == 1 and results:
            return results
        or_q = " OR ".join(_fts_quote(t) + ("*" if t == last and len(t) >= 3 else "") for t in selective)
        seen = {d.rowid for d, _ in results}
        return results + [(d, s) for d, s in self.store.bm25(or_q, languages, pool) if d.rowid not in seen]

    # ------------------------------------------------------------------
    def correct(self, q: QueryText) -> QueryText | None:
        """Spelling-correct content tokens that do not exist in the vocabulary."""
        df = self.store.vocab_df
        toks = tokens(q.folded)
        changed = False
        out = []
        for t in toks:
            if len(t) < 4 or t in df or not t.isalpha():
                out.append(t)
                continue
            cands = process.extract(t, self.store.vocab_near_length(len(t)), scorer=fuzz.ratio,
                                    score_cutoff=self.fuzzy_min_similarity, limit=5)
            if cands:
                # best similarity; ties -> the more frequent term
                best = max(cands, key=lambda c: (round(c[1], 1), df.get(c[0], 0)))[0]
                out.append(best)
                changed = True
            else:
                out.append(t)
        if not changed:
            return None
        return QueryText.build(" ".join(out), source="spelling_correction", weight=q.weight)

    @staticmethod
    def _score(q: QueryText, hits: dict[int, DescriptionHit]) -> None:
        max_bm25 = max((h.bm25 for h in hits.values()), default=0.0) or 1.0
        qtoks = list(q.content) or tokens(q.folded)
        for h in hits.values():
            d = h.description
            sim = fuzz.token_sort_ratio(q.folded, d.folded) / 100.0
            cov = coverage(qtoks, content_tokens(d.folded) or tokens(d.folded))
            h.lexical = max(h.exact, 0.5 * sim + 0.3 * cov + 0.2 * (h.bm25 / max_bm25))
