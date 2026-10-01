import json

import httpx

from clif.reranking.base import Candidate, RerankRequest
from clif.reranking.llm import LocalLLMReranker


def _req():
    cands = [Candidate("A", "74400008", "Appendicitis", "Appendicitis (disorder)", "disorder"),
             Candidate("B", "85189001", "Acute appendicitis", "Acute appendicitis (disorder)", "disorder")]
    return RerankRequest("appendicitis", "appendicitis", None, cands)


def test_llm_constrained_choice_with_logprobs():
    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["choice"]["enum"] == ["A", "B", "NO_MATCH", "AMBIGUOUS"]
        content = '{"choice": "A", "confidence": 0.9, "reason": "no acuity stated"}'
        toks = [{"token": '{"choice": "', "logprob": 0.0},
                {"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "A", "logprob": -0.1},
                                                                 {"token": "B", "logprob": -2.5},
                                                                 {"token": "NO", "logprob": -5.0}]},
                {"token": '", "confidence": 0.9, "reason": "no acuity stated"}', "logprob": 0.0}]
        return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                      "logprobs": {"content": toks}}]})

    rr = LocalLLMReranker("http://llm/v1", "local-model", client=httpx.Client(transport=httpx.MockTransport(handler)))
    res = rr.rerank(_req())
    assert res.scores_are_probabilities
    assert res.scores["74400008"] > 0.85 and res.scores["85189001"] > 0.05
    assert res.chosen == "74400008"


def test_llm_invalid_choice_never_accepted():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {
            "content": '{"choice": "22298006", "confidence": 1.0, "reason": "x"}'}}]})

    rr = LocalLLMReranker("http://llm/v1", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    res = rr.rerank(_req())
    assert res.chosen == "NO_MATCH" and "22298006" in res.rejected
