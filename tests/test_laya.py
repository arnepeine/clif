"""Laya adapter tests against a mock laya-serve speaking POST /v1/systemone."""

import json

import httpx
import pytest

from clif.models.schemas import MapRequest
from clif.pipeline import ClifMapper
from clif.reranking.base import Candidate, RerankerError, RerankRequest
from clif.reranking.laya import QUESTION_ID, LayaHttpReranker, build_laya_request, parse_laya_response


def fake_laya(collapse: bool = False, extra_label: str | None = None, seen: list | None = None):
    """Mimics laya-serve: gives most mass to the option whose label equals the clinical text."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone"
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        q = body["questions"][QUESTION_ID]
        assert q["type"] == "choice"
        labels = list(q["criteria"])
        text = body["state"]["clinical_text"].lower()
        gold = next((lb for lb in labels if lb.lower() == text), "NO_MATCH")
        rest = 0.1 / (len(labels) - 1)
        probs = {lb: (0.9 if lb == gold else rest) for lb in labels}
        if extra_label:
            probs[extra_label] = 0.5
        usage = {"input_tokens": 120, "output_tokens": 0}
        if collapse:
            usage["options"] = {QUESTION_ID: {"total": len(labels), "distinct": len(labels) - 2,
                                              "tokens_per_option": 4}}
        return httpx.Response(200, json={
            "model": "laya-rl-agent",
            "answers": {QUESTION_ID: {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs,
                                      "confidence": 0.7, "answer_confidence": probs[gold]}},
            "usage": usage,
            "routing": {"model": body.get("model", "english"), "reason": "test"},
        })

    return handler


def laya_reranker(**kw) -> LayaHttpReranker:
    seen = kw.pop("seen", None)
    client = httpx.Client(transport=httpx.MockTransport(fake_laya(seen=seen, **kw)), base_url="http://laya:8000")
    return LayaHttpReranker("http://laya:8000", client=client)


def _mapper_with(mapper, reranker):
    return ClifMapper(mapper.settings, mapper.store, mapper.retriever, reranker, mapper.analyzer, mapper.policy)


def _req(pts: list[str], core="appendicitis", language="en") -> RerankRequest:
    cands = [Candidate(label=chr(65 + i), concept_id=str(100000 + i), preferred_term=pt, fsn=f"{pt} (disorder)",
                       semantic_tag="disorder", synonyms=[f"{pt} syn1", f"{pt} syn2", f"{pt} syn3"])
             for i, pt in enumerate(pts)]
    return RerankRequest(text=core, core_text=core, context=None, candidates=cands, language=language)


def test_request_uses_preferred_terms_as_labels():
    body, label_map = build_laya_request(_req(["Appendicitis", "Acute appendicitis"]))
    crit = body["questions"][QUESTION_ID]["criteria"]
    assert list(crit)[:2] == ["Appendicitis", "Acute appendicitis"]
    assert crit["Appendicitis"] == "disorder; also: Appendicitis syn1, Appendicitis syn2"
    assert {"NO_MATCH", "AMBIGUOUS"} <= set(crit)
    assert label_map["Appendicitis"] == "A"
    assert body["state"] == {"clinical_text": "appendicitis"}  # only the core text, no negation cues
    assert body["model"] == "english"


def test_boolean_and_duplicate_labels_are_disambiguated():
    body, _ = build_laya_request(_req(["Yes", "Cold", "Cold"]))
    labels = list(body["questions"][QUESTION_ID]["criteria"])
    assert labels[0] == "Yes (disorder)"  # boolean words confuse Laya (see its Honest limits)
    assert labels[1] == "Cold" and labels[2] != "Cold"
    assert len(set(labels)) == len(labels)


def test_head_budget_grows_with_options():
    small, _ = build_laya_request(_req(["A b"] * 1))
    big, _ = build_laya_request(_req([f"Long clinical concept name number {i}" for i in range(32)]))
    assert small["head_max_len"] == 256
    assert big["head_max_len"] > 256 and big["head_max_len"] <= 1024
    assert big["max_len"] == big["head_max_len"] + 512


def test_language_selects_checkpoint():
    assert build_laya_request(_req(["Herzinfarkt"], language="de"))[0]["model"] == "multilingual"
    assert "model" not in build_laya_request(_req(["x"], language=None))[0]


def test_parse_maps_back_and_rejects_unknown():
    _, label_map = build_laya_request(_req(["Appendicitis"]))
    scores, meta = parse_laya_response({"answers": {QUESTION_ID: {"probabilities": {"Appendicitis": 0.8,
                                                                                    "NO_MATCH": 0.1,
                                                                                    "Invented": 0.1}}}}, label_map)
    assert scores["A"] == 0.8 and scores["NO_MATCH"] == 0.1 and "unknown:Invented" in scores
    with pytest.raises(RerankerError):
        parse_laya_response({"answers": {}}, label_map)


def test_end_to_end_with_laya(mapper):
    seen: list = []
    m = _mapper_with(mapper, laya_reranker(seen=seen))
    r = m.map(MapRequest(text="patient denies chest pain", debug=True))
    assert r.status == "MATCH" and r.concept.concept_id == "29857009" and r.assertion == "NEGATED"
    assert r.reranker.backend == "laya"
    assert seen[0]["state"]["clinical_text"] == "chest pain"  # negation stripped before Laya sees it
    assert len(seen[0]["questions"][QUESTION_ID]["criteria"]) <= mapper.settings.rerank_candidates + 2
    assert r.debug["reranker"]["meta"]["laya_answer_confidence"] == pytest.approx(0.9)


def test_laya_no_match(mapper):
    r = _mapper_with(mapper, laya_reranker()).map(MapRequest(text="parking garage level three"))
    assert r.status == "NO_MATCH"


def test_option_collapse_forces_ambiguous(mapper):
    r = _mapper_with(mapper, laya_reranker(collapse=True)).map(MapRequest(text="chest pain"))
    assert r.status == "AMBIGUOUS" and "option_collapse" in r.reason


def test_invented_label_is_rejected(mapper):
    r = _mapper_with(mapper, laya_reranker(extra_label="Totally new concept")).map(
        MapRequest(text="chest pain", debug=True))
    assert "unknown:Totally new concept" in r.debug["reranker"]["rejected"]
    assert r.concept is None or r.concept.concept_id == "29857009"


def test_server_error_raises(mapper):
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500, json={"detail": "x"})))
    rr = LayaHttpReranker("http://laya:8000", client=client)
    with pytest.raises(RerankerError):
        _mapper_with(mapper, rr).map(MapRequest(text="chest pain"))


def test_fallback_to_heuristic(mapper):
    from clif.reranking import FallbackReranker, HeuristicReranker

    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500)))
    rr = FallbackReranker(LayaHttpReranker("http://laya:8000", client=client), HeuristicReranker())
    r = _mapper_with(mapper, rr).map(MapRequest(text="chest pain"))
    assert r.status == "MATCH" and r.reranker.fallback_used and "reranker_fallback_used" in r.warnings
