from __future__ import annotations

import logging

from clif.config import Settings
from clif.reranking.base import (
    AMBIGUOUS,
    NO_MATCH,
    Candidate,
    ConceptReranker,
    RerankerError,
    RerankRequest,
    RerankResult,
)
from clif.reranking.heuristic import HeuristicReranker

log = logging.getLogger(__name__)

__all__ = [
    "AMBIGUOUS", "NO_MATCH", "Candidate", "ConceptReranker", "RerankRequest", "RerankResult", "RerankerError",
    "HeuristicReranker", "FallbackReranker", "make_reranker",
]


class FallbackReranker(ConceptReranker):
    """Uses `fallback` when `primary` fails; the result is flagged."""

    def __init__(self, primary: ConceptReranker, fallback: ConceptReranker):
        self.primary, self.fallback = primary, fallback
        self.name, self.model = primary.name, primary.model

    def rerank(self, request: RerankRequest) -> RerankResult:
        try:
            return self.primary.rerank(request)
        except RerankerError as e:
            log.warning("primary reranker failed, using fallback: %s", e)
            res = self.fallback.rerank(request)
            res.fallback_used = True
            return res


def make_reranker(settings: Settings) -> ConceptReranker:
    if settings.reranker == "heuristic":
        return HeuristicReranker()
    laya_opts = dict(model=settings.laya_model, option_synonyms=settings.laya_option_synonyms,
                     head_min=settings.laya_head_max_len_min, head_max=settings.laya_head_max_len_max,
                     state_budget=settings.laya_state_budget)
    if settings.reranker == "laya":
        from clif.reranking.laya import LayaHttpReranker

        primary: ConceptReranker = LayaHttpReranker(settings.laya_url, settings.laya_api_key,
                                                    settings.reranker_timeout_s, **laya_opts)
    elif settings.reranker == "laya-local":
        from clif.reranking.laya import LayaLocalReranker

        primary = LayaLocalReranker(settings.laya_checkpoint, settings.laya_device, **laya_opts)
    elif settings.reranker == "llm":
        from clif.reranking.llm import LocalLLMReranker

        primary = LocalLLMReranker(settings.llm_url, settings.llm_model, settings.llm_api_key,
                                   settings.reranker_timeout_s, settings.llm_use_json_schema)
    else:
        raise ValueError(f"unknown reranker {settings.reranker}")
    if settings.reranker_fallback == "heuristic":
        return FallbackReranker(primary, HeuristicReranker())
    return primary
