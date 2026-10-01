"""The mapping pipeline.

text -> normalization + context analysis (assertion, temporality, abbreviations)
     -> soft semantic-category detection
     -> hybrid SNOMED-wide retrieval (top ~32 concepts)
     -> constrained-choice reranking (candidates + NO_MATCH + AMBIGUOUS)
     -> terminology validation (active, offered, hierarchy/ECL)
     -> calibrated decision: MATCH | AMBIGUOUS | NO_MATCH | INSUFFICIENT_INFORMATION
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from clif import __version__
from clif.clinical_context.analyzer import ContextAnalysis, ContextAnalyzer
from clif.clinical_context.language import detect_language
from clif.config import Settings
from clif.decision import DecisionPolicy, decide
from clif.models.schemas import (
    Alternative,
    BatchItemError,
    CandidateOut,
    CandidatesRequest,
    CandidatesResponse,
    ConceptOut,
    MapRequest,
    MapResponse,
    MatchedDescriptionOut,
    RerankerInfo,
)
from clif.reranking import ConceptReranker, RerankerError, make_reranker
from clif.reranking.base import Candidate, RerankRequest, RerankResult, option_labels
from clif.retrieval.category import CategoryPrediction, predict_category
from clif.retrieval.hybrid import HybridRetriever, RetrievalResult, RetrievedConcept
from clif.retrieval.vector import CachedEmbedder, VectorIndex, make_embedder
from clif.terminology.hierarchy import EclUnsupported, HierarchyConstraint
from clif.terminology.store import TerminologyStore
from clif.text import QueryText

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    request_text: str
    analysis: ContextAnalysis
    variants: list[QueryText]
    source_language: str
    language_confident: bool
    languages: list[str] | None
    category: CategoryPrediction
    constraint: HierarchyConstraint | None
    exclude_tags: frozenset[str]
    warnings: list[str]


class ClifMapper:
    def __init__(self, settings: Settings, store: TerminologyStore, retriever: HybridRetriever,
                 reranker: ConceptReranker, analyzer: ContextAnalyzer, policy: DecisionPolicy):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.reranker = reranker
        self.analyzer = analyzer
        self.policy = policy

    @classmethod
    def from_settings(cls, settings: Settings, reranker: ConceptReranker | None = None) -> "ClifMapper":
        store = TerminologyStore(settings.db_path)
        vector_index = embedder = None
        if settings.embedder != "none" and (settings.vector_dir / "meta.json").exists():
            vector_index = VectorIndex(settings.vector_dir)
            base = make_embedder(settings)
            if base is not None and base.identity != vector_index.identity:
                raise RuntimeError(
                    f"vector index was built with {vector_index.identity} but CLIF_EMBEDDER is {base.identity}; "
                    f"rebuild the index or fix the configuration"
                )
            embedder = CachedEmbedder(base) if base else None
        retriever = HybridRetriever(store, settings, vector_index, embedder)
        analyzer = ContextAnalyzer.default(settings.situation_mode, settings.abbreviation_dominance)
        return cls(settings, store, retriever, reranker or make_reranker(settings), analyzer,
                   DecisionPolicy.from_settings(settings))

    # ------------------------------------------------------------------ prepare
    def prepare(self, text: str, context: str | None, language: str,
                allowed_hierarchies: list[str] | None, ecl: str | None) -> Prepared:
        warnings: list[str] = []
        analysis = self.analyzer.analyze(text, context)
        warnings.extend(analysis.warnings)
        if language and language != "auto":
            src, confident = language.lower(), True
        else:
            src, confident = detect_language(analysis.core or text)
        if confident:
            languages = list(dict.fromkeys([src, *self.settings.fallback_languages]))
            if src not in self.store.languages:
                warnings.append(f"no indexed descriptions for language '{src}'; cross-lingual fallback only")
        else:
            languages = None  # search every indexed language
        include_full = self.settings.situation_mode == "prefer" and analysis.has_context_cues
        variants = self.analyzer.query_variants(analysis, include_full_text=include_full)
        category = predict_category(analysis.original if analysis.fixed_expression else analysis.core)
        constraint = HierarchyConstraint.build(allowed_hierarchies, ecl)
        if constraint and constraint.ecl is not None and not self.store.has_hierarchy:
            raise EclUnsupported("ECL needs IS-A relationships, which this release does not contain "
                                 "(e.g. the Global Patient Set); use allowed_hierarchies instead")
        # Context was split off into assertion fields; offering precoordinated
        # situation concepts as well would encode it twice.
        exclude = frozenset()
        if analysis.has_context_cues and not analysis.fixed_expression and self.settings.situation_mode != "prefer":
            exclude = frozenset({"situation"})
        return Prepared(text, analysis, variants, src, confident, languages, category, constraint, exclude, warnings)

    def _retrieve(self, p: Prepared, limit: int | None = None) -> RetrievalResult:
        return self.retriever.retrieve(p.variants, p.languages, limit or self.settings.candidate_count,
                                       p.constraint, p.category, p.exclude_tags)

    # --------------------------------------------------------------- candidates
    def candidates(self, req: CandidatesRequest) -> CandidatesResponse:
        t0 = time.perf_counter()
        p = self.prepare(req.text, req.context, req.language, req.allowed_hierarchies, req.ecl)
        r = self._retrieve(p, req.limit) if not p.analysis.insufficient else None
        return CandidatesResponse(
            input=req.text,
            core_text=p.analysis.core,
            queries=[{"text": v.text, "normalized": v.norm, "folded": v.folded, "source": v.source,
                      "weight": round(v.weight, 3)} for v in p.variants],
            source_language=p.source_language,
            searched_languages=p.languages,
            context_analysis=_analysis_dict(p.analysis),
            semantic_category=_category_dict(p.category),
            candidates=[_candidate_out(c) for c in (r.candidates if r else [])],
            filtered=r.filtered if r else [],
            descriptions_considered=r.descriptions_considered if r else 0,
            concepts_considered=r.concepts_considered if r else 0,
            channel_counts=r.channel_counts if r else {},
            vector_enabled=self.retriever.vector_enabled,
            snomed_version=self.store.snomed_version,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ---------------------------------------------------------------------- map
    def _rerank_request(self, p: Prepared, context: str | None, r: RetrievalResult) -> RerankRequest:
        pool = r.candidates[: self.settings.rerank_candidates]
        labels = option_labels(len(pool))
        cands = []
        for label, rc in zip(labels, pool):
            rec = self.store.concept(rc.concept_id)
            syn_langs = list(dict.fromkeys(["en", p.source_language]))
            synonyms = [s for lang in syn_langs for s in (rec.synonyms.get(lang, []) if rec else [])][:6]
            parents = []
            for pid in (rec.parents if rec else [])[:3]:
                prec = self.store.concept(pid)
                if prec:
                    parents.append(prec.preferred_term("en")[0])
            cands.append(Candidate(
                label=label, concept_id=rc.concept_id, preferred_term=rc.preferred_term, fsn=rc.fsn,
                semantic_tag=rc.semantic_tag, synonyms=synonyms, hierarchy=parents,
                matched_terms=list(dict.fromkeys(m.term for m in rc.matched)),
                retrieval_score=round(rc.score, 4), retrieval_rank=rc.rank,
            ))
        expanded = [v.text for v in p.variants if v.source == "abbreviation"]
        language = p.source_language if p.language_confident else None
        if language is None and pool:
            # Short phrases rarely reveal their language; the description the top
            # candidate matched is better evidence (it picks Laya's checkpoint).
            language = pool[0].best.language
        return RerankRequest(text=p.request_text, core_text=p.analysis.core or p.request_text, context=context,
                             candidates=cands, expanded_forms=expanded, language=language)

    def map(self, req: MapRequest, _rerank_result: RerankResult | None = None,
            _prepared: tuple[Prepared, RetrievalResult] | None = None) -> MapResponse:
        t0 = time.perf_counter()
        self._check_lengths(req.text, req.context)
        if _prepared:
            p, r = _prepared
        else:
            p = self.prepare(req.text, req.context, req.language, req.allowed_hierarchies, req.ecl)
            r = None
        display_lang = req.display_language or self.settings.default_display_language
        a = p.analysis
        base = dict(
            input=req.text, context_supplied=bool(req.context), assertion=a.assertion.value,
            temporality=a.temporality.value, certainty=a.certainty.value, experiencer=a.experiencer.value,
            source_language=p.source_language, display_language=display_lang,
            semantic_category_hint=p.category.top if p.category.certainty >= 0.3 else None,
            snomed_version=self.store.snomed_version, snomed_edition=self.store.snomed_edition,
            reranker=RerankerInfo(backend=self.reranker.name, model=self.reranker.model),
            calibrated=self.policy.calibrator.fitted, clif_version=__version__, warnings=list(p.warnings),
        )
        debug: dict | None = {} if req.debug else None
        if debug is not None:
            debug.update(
                normalized_query=[{"text": v.text, "normalized": v.norm, "source": v.source,
                                   "weight": round(v.weight, 3)} for v in p.variants],
                context_analysis=_analysis_dict(a),
                language={"source": p.source_language, "confident": p.language_confident,
                          "searched": p.languages},
                semantic_category=_category_dict(p.category),
                constraint=p.constraint.describe() if p.constraint else None,
            )

        if a.insufficient:
            if debug is not None:
                debug["timing_ms"] = {"total": round((time.perf_counter() - t0) * 1000, 2)}
            return MapResponse(status="INSUFFICIENT_INFORMATION",
                               reason="no concept-bearing text after removing context cues and fillers",
                               debug=debug, **base)

        if r is None:
            r = self._retrieve(p)
        if debug is not None:
            debug["retrieval"] = {
                "elapsed_ms": round(r.elapsed_ms, 2), "cached": r.cached, "vector_enabled": r.vector_enabled,
                "descriptions_considered": r.descriptions_considered, "concepts_considered": r.concepts_considered,
                "channel_counts": r.channel_counts,
                "candidates": [_candidate_out(c).model_dump() for c in r.candidates],
                "filtered": r.filtered[:50],
            }
        if not r.candidates:
            reason = "no candidates retrieved"
            if p.constraint and r.filtered:
                reason = "no retrieved candidate satisfies the hierarchy/ECL restriction"
            if debug is not None:
                debug["timing_ms"] = {"total": round((time.perf_counter() - t0) * 1000, 2)}
            return MapResponse(status="NO_MATCH", reason=reason, debug=debug, **base)

        rreq = self._rerank_request(p, req.context, r)
        t_rr = time.perf_counter()
        rres = _rerank_result or self.reranker.rerank(rreq)
        rerank_ms = (time.perf_counter() - t_rr) * 1000
        base["reranker"] = RerankerInfo(backend=rres.backend, model=rres.model, fallback_used=rres.fallback_used)
        if rres.fallback_used:
            base["warnings"].append("reranker_fallback_used")

        # ---- terminology validation --------------------------------------
        offered = {c.concept_id: c for c in rreq.candidates}
        by_id: dict[str, RetrievedConcept] = {c.concept_id: c for c in r.candidates}
        checks: list[dict] = []
        valid_scores: dict[str, float] = {}
        for cid, score in rres.scores.items():
            problems = []
            if cid not in offered:
                problems.append("not_in_candidate_set")
            elif not self.store.is_active(cid):
                problems.append("inactive_concept")
            elif p.constraint is not None:
                ok, why = p.constraint.allows(cid, self.store)
                if not ok:
                    problems.append(why)
            if problems:
                checks.append({"concept_id": cid, "passed": False, "problems": problems})
            else:
                valid_scores[cid] = score
        if rres.rejected:
            checks.append({"rejected_labels": rres.rejected, "passed": False,
                           "problems": ["reranker returned options that were not supplied"]})

        force_reason = None
        unresolved = a.unresolved_abbreviations
        if unresolved:
            force_reason = "ambiguous_abbreviation: " + "; ".join(
                f"{h.abbreviation} = " + " | ".join(f"{e} ({pp:.2f})" for e, pp in h.plausible) for h in unresolved
            )
        elif "option_collapse" in rres.warnings:
            force_reason = "reranker_option_collapse: candidate labels were not distinguishable by the model"

        dec = decide(self.policy, valid_scores, rres.no_match, rres.ambiguous,
                     {cid: by_id[cid].score for cid in valid_scores if cid in by_id}, force_reason)

        # Selected concept must have been matched through an active description.
        selected = dec.top_concept if dec.status == "MATCH" else None
        concept_out = None
        matched_lang = None
        if selected:
            rc = by_id[selected]
            rec = self.store.concept(selected)
            best = rc.best
            desc_ok = self._description_active(best.description_id)
            checks.append({"concept_id": selected, "passed": desc_ok and bool(rec and rec.active),
                           "checks": ["active_concept", "in_candidate_set", "active_matched_description",
                                      *(["hierarchy_constraint"] if p.constraint else [])]})
            if not desc_ok:  # defensive: the index only holds active descriptions
                dec.status, dec.reason, selected = "NO_MATCH", "matched description is inactive", None
            else:
                pt, _ = rec.preferred_term(display_lang)  # type: ignore[union-attr]
                concept_out = ConceptOut(
                    concept_id=selected, preferred_term=pt, fsn=rc.fsn, semantic_tag=rc.semantic_tag,
                    matched_description=MatchedDescriptionOut(description_id=best.description_id, term=best.term,
                                                              language=best.language, type=best.type),
                )
                matched_lang = best.language

        alternatives: list[Alternative] = []
        for cid, conf in sorted(dec.calibrated_scores.items(), key=lambda kv: (-kv[1], kv[0])):
            if cid == selected or len(alternatives) >= req.top_k:
                continue
            rc = by_id[cid]
            rec = self.store.concept(cid)
            alternatives.append(Alternative(
                concept_id=cid, preferred_term=rec.preferred_term(display_lang)[0] if rec else rc.preferred_term,
                semantic_tag=rc.semantic_tag, confidence=round(conf, 4),
                reranker_score=round(rres.scores.get(cid, 0.0), 4), retrieval_score=round(rc.score, 4),
                retrieval_rank=rc.rank,
            ))
        if matched_lang is None and alternatives:
            matched_lang = by_id[alternatives[0].concept_id].best.language
        if not p.language_confident and matched_lang:
            # Short phrases rarely reveal their language; the description the
            # text actually matched is better evidence than the detector.
            base["source_language"] = matched_lang
            if debug is not None:
                debug["language"]["source"] = matched_lang
                debug["language"]["inferred_from_matched_description"] = True

        if debug is not None:
            debug["reranker"] = {
                "backend": rres.backend, "model": rres.model, "latency_ms": round(rerank_ms, 2),
                "candidate_count": len(rreq.candidates),
                "probabilities": {c.concept_id: round(rres.scores.get(c.concept_id, 0.0), 4)
                                  for c in rreq.candidates},
                "no_match": round(rres.no_match, 4), "ambiguous": round(rres.ambiguous, 4),
                "scores_are_probabilities": rres.scores_are_probabilities, "explanation": rres.explanation,
                "rejected": rres.rejected, "meta": rres.meta,
            }
            debug["decision"] = {
                "status": dec.status, "reason": dec.reason, "top_concept": dec.top_concept,
                "confidence": round(dec.confidence, 4), "margin": round(dec.margin, 4),
                "thresholds": asdict(self.policy.thresholds), "policy_source": self.policy.source,
            }
            if dec.top_concept and dec.top_concept in by_id:
                top = by_id[dec.top_concept]
                debug["selected"] = {"concept_id": dec.top_concept, "retrieval_rank": top.rank,
                                     "retrieval_score": round(top.score, 4),
                                     "reranker_score": round(rres.scores.get(dec.top_concept, 0.0), 4)}
            debug["validation"] = {"passed": all(c.get("passed", True) for c in checks
                                                 if c.get("concept_id") == selected) if selected else None,
                                   "checks": checks}
            debug["timing_ms"] = {"retrieval": round(r.elapsed_ms, 2), "rerank": round(rerank_ms, 2),
                                  "total": round((time.perf_counter() - t0) * 1000, 2)}

        return MapResponse(
            status=dec.status,  # type: ignore[arg-type]
            concept=concept_out,
            # Confidence refers to the returned concept; without one it is 0 and the
            # alternatives carry their own confidences.
            confidence=round(dec.confidence, 4) if dec.status == "MATCH" else 0.0,
            alternatives=alternatives,
            reason=dec.reason,
            matched_description_language=matched_lang,
            debug=debug,
            **base,
        )

    def map_batch(self, reqs: list[MapRequest]) -> list[MapResponse | BatchItemError]:
        """Bulk mapping: one batched embedding call, shared retrieval cache, and
        either a batched reranker call (Laya in-process) or bounded concurrency."""
        preps: list[Prepared | Exception] = []
        for rq in reqs:
            try:
                self._check_lengths(rq.text, rq.context)
                preps.append(self.prepare(rq.text, rq.context, rq.language, rq.allowed_hierarchies, rq.ecl))
            except (ValueError, RuntimeError) as e:
                preps.append(e)
        if self.retriever.embedder is not None:  # one batched embedding call for all queries
            texts = list({v.text for p in preps if not isinstance(p, Exception) for v in p.variants})
            if texts:
                self.retriever.embedder.embed_many(texts)
        prepared: list[tuple[Prepared, RetrievalResult | None] | Exception] = []
        for p in preps:
            if isinstance(p, Exception):
                prepared.append(p)
            else:
                prepared.append((p, None if p.analysis.insufficient else self._retrieve(p)))

        rerank_many = getattr(self.reranker, "rerank_many", None)
        pre_rr: dict[int, RerankResult] = {}
        if rerank_many is not None:
            idx = [i for i, x in enumerate(prepared)
                   if not isinstance(x, Exception) and x[1] is not None and x[1].candidates]
            if idx:
                try:
                    reqs_rr = [self._rerank_request(prepared[i][0], reqs[i].context, prepared[i][1]) for i in idx]
                    pre_rr = dict(zip(idx, rerank_many(reqs_rr)))
                except RerankerError as e:
                    log.warning("batched rerank failed, falling back to per-item: %s", e)

        def one(i: int) -> MapResponse | BatchItemError:
            x = prepared[i]
            if isinstance(x, Exception):
                return BatchItemError(input=reqs[i].text, error=str(x))
            p, r = x
            try:
                return self.map(reqs[i], _rerank_result=pre_rr.get(i),
                                _prepared=(p, r) if r is not None else None)
            except (RerankerError, ValueError, RuntimeError) as e:
                return BatchItemError(input=reqs[i].text, error=str(e))

        with ThreadPoolExecutor(max_workers=max(1, self.settings.reranker_concurrency)) as ex:
            return list(ex.map(one, range(len(reqs))))

    # ------------------------------------------------------------------ helpers
    def _check_lengths(self, text: str, context: str | None) -> None:
        if len(text) > self.settings.max_text_length:
            raise ValueError(f"text longer than {self.settings.max_text_length} characters")
        if context and len(context) > self.settings.max_context_length:
            raise ValueError(f"context longer than {self.settings.max_context_length} characters")

    def _description_active(self, description_id: str) -> bool:
        row = self.store._conn().execute(
            "SELECT d.active, c.active FROM descriptions d JOIN concepts c ON c.id = d.concept_id WHERE d.id = ?",
            (description_id,),
        ).fetchone()
        return bool(row and row[0] and row[1])


def _analysis_dict(a: ContextAnalysis) -> dict:
    return {
        "core_text": a.core,
        "fixed_expression": a.fixed_expression,
        "assertion": a.assertion.value,
        "temporality": a.temporality.value,
        "certainty": a.certainty.value,
        "experiencer": a.experiencer.value,
        "cues": [{"category": c.category, "cue": c.cue, "span": [c.start, c.end]} for c in a.cues],
        "abbreviations": [
            {"abbreviation": h.abbreviation, "resolved": h.resolved,
             "expansions": [{"expansion": e, "posterior": round(pp, 3), "context_cues": h.cue_hits.get(e, [])}
                            for e, pp in h.posteriors]}
            for h in a.abbreviations
        ],
        "warnings": a.warnings,
    }


def _category_dict(c: CategoryPrediction) -> dict:
    top = sorted(c.probs.items(), key=lambda kv: -kv[1])[:4]
    return {"top": c.top, "certainty": round(c.certainty, 3), "distribution": {k: round(v, 3) for k, v in top},
            "evidence": c.evidence}


def _candidate_out(c: RetrievedConcept) -> CandidateOut:
    return CandidateOut(
        rank=c.rank, concept_id=c.concept_id, preferred_term=c.preferred_term, fsn=c.fsn,
        semantic_tag=c.semantic_tag, retrieval_score=round(c.score, 4), lexical_score=round(c.lexical, 4),
        semantic_score=round(c.semantic, 4), exact_or_synonym_bonus=round(c.exact, 4),
        terminology_prior=round(c.prior, 3), category_factor=round(c.category_factor, 3),
        matched_descriptions=[{"description_id": m.description_id, "term": m.term, "language": m.language,
                               "type": m.type, "channels": m.channels, "lexical": m.lexical,
                               "semantic": m.semantic, "query": m.variant} for m in c.matched],
    )
