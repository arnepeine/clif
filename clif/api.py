"""REST API.

    uvicorn clif.api:app --host 0.0.0.0 --port 8000

POST /map          map one phrase
POST /map/batch    map many phrases (batched embeddings, batched/concurrent reranking)
POST /candidates   retrieval only, no reranking (debugging)
GET  /concepts/{id} concept record with parents/ancestors
GET  /health       readiness incl. SNOMED version and reranker backend
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from clif import __version__
from clif.config import Settings, get_settings
from clif.models.schemas import (
    BatchMapRequest,
    BatchMapResponse,
    CandidatesRequest,
    CandidatesResponse,
    MapRequest,
    MapResponse,
)
from clif.pipeline import ClifMapper
from clif.reranking import RerankerError
from clif.terminology.hierarchy import EclError

log = logging.getLogger("clif.api")


class _State:
    mapper: ClifMapper | None = None
    error: str | None = None
    lock = threading.Lock()


def _load(settings: Settings) -> ClifMapper | None:
    with _State.lock:
        if _State.mapper is None:
            try:
                _State.mapper = ClifMapper.from_settings(settings)
                _State.error = None
                log.info("Clif ready: SNOMED %s, reranker %s", _State.mapper.store.snomed_version,
                         _State.mapper.reranker.name)
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                _State.error = str(e)
                log.warning("Clif not ready: %s", e)
        return _State.mapper


def create_app(settings: Settings | None = None, mapper: ClifMapper | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if mapper is not None:
            _State.mapper = mapper
        else:
            await run_in_threadpool(_load, settings)
        yield

    app = FastAPI(title="Clif", version=__version__, lifespan=lifespan,
                  description="Clinical free text to SNOMED CT: SNOMED-wide retrieval, constrained-choice reranking.")

    def ready() -> ClifMapper:
        m = _State.mapper or _load(settings)
        if m is None:
            raise HTTPException(status_code=503, detail=f"terminology index not ready: {_State.error}")
        return m

    @app.get("/health")
    def health():
        m = _State.mapper
        if m is None:
            return {"status": "not_ready", "error": _State.error, "clif_version": __version__}
        return {
            "status": "ok",
            "clif_version": __version__,
            "snomed_version": m.store.snomed_version,
            "snomed_edition": m.store.snomed_edition,
            "languages": m.store.languages,
            "reranker": {"backend": m.reranker.name, "model": m.reranker.model},
            "vector_retrieval": m.retriever.vector_enabled,
            "calibrated": m.policy.calibrator.fitted,
        }

    @app.post("/map", response_model=MapResponse, response_model_exclude_none=False)
    async def map_one(req: MapRequest) -> MapResponse:
        m = ready()
        try:
            res = await run_in_threadpool(m.map, req)
        except (ValueError, EclError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RerankerError as e:
            raise HTTPException(status_code=503, detail=f"reranker unavailable: {e}")
        if not req.debug:
            res.debug = None
        return res

    @app.post("/map/batch", response_model=BatchMapResponse)
    async def map_batch(req: BatchMapRequest) -> BatchMapResponse:
        m = ready()
        if len(req.items) > settings.max_batch_size:
            raise HTTPException(status_code=413, detail=f"batch larger than {settings.max_batch_size} items")
        t0 = time.perf_counter()
        results = await run_in_threadpool(m.map_batch, req.items)
        return BatchMapResponse(results=results, elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))

    @app.post("/candidates", response_model=CandidatesResponse)
    async def candidates(req: CandidatesRequest) -> CandidatesResponse:
        m = ready()
        try:
            return await run_in_threadpool(m.candidates, req)
        except (ValueError, EclError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/concepts/{concept_id}")
    def concept(concept_id: str, language: str = "en"):
        m = ready()
        rec = m.store.full_record(concept_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown concept")
        pt, pt_lang = rec.preferred_term(language)
        return {
            "concept_id": rec.concept_id, "active": rec.active, "fsn": rec.fsn, "preferred_term": pt,
            "preferred_term_language": pt_lang, "preferred_terms": rec.preferred_terms,
            "synonyms": rec.synonyms, "semantic_tag": rec.semantic_tag, "parents": rec.parents,
            "ancestors": rec.ancestors, "snomed_version": m.store.snomed_version,
        }

    return app


app = create_app()
