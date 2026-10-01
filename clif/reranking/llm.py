"""Generic local LLM reranker via an OpenAI-compatible chat completions API.

Works with on-prem servers such as vLLM, llama.cpp server, TGI or Ollama.
Output is constrained with a JSON schema whose ``choice`` field is an enum of
the option labels, so the model can only pick a supplied option. When the
server returns token logprobs, the probability distribution over options is
read from the first token of the ``choice`` value; otherwise the model's
self-reported confidence is used and flagged as not a probability.
"""

from __future__ import annotations

import json
import math
import re
import time

import httpx

from clif.reranking.base import (
    ConceptReranker,
    RerankerError,
    RerankRequest,
    RerankResult,
    build_choice_question,
    result_from_label_scores,
)

SYSTEM_PROMPT = (
    "You are a SNOMED CT terminology mapping component. You do not diagnose. "
    "You receive clinical text and a closed list of options. Reply only with JSON: "
    '{"choice": <option label>, "confidence": <0..1>, "reason": <max 20 words>}.'
)


class LocalLLMReranker(ConceptReranker):
    name = "llm"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 30.0,
                 use_json_schema: bool = True, client: httpx.Client | None = None):
        if not base_url or not model:
            raise ValueError("CLIF_LLM_URL and CLIF_LLM_MODEL are required for the llm reranker")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.use_json_schema = use_json_schema
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

    def _payload(self, request: RerankRequest) -> tuple[dict, list[str]]:
        q = build_choice_question(request)
        labels = list(q["question"]["options"].keys())
        user = (
            q["question"]["instructions"]
            + "\n\nINPUT:\n"
            + json.dumps(q["state"], ensure_ascii=False)
            + "\n\nOPTIONS:\n"
            + json.dumps(q["question"]["options"], ensure_ascii=False, indent=1)
        )
        body: dict = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 120,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "logprobs": True,
            "top_logprobs": 20,
        }
        if self.use_json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "snomed_choice",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "choice": {"type": "string", "enum": labels},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["choice", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                },
            }
        else:
            body["response_format"] = {"type": "json_object"}
        return body, labels

    def rerank(self, request: RerankRequest) -> RerankResult:
        t0 = time.perf_counter()
        body, labels = self._payload(request)
        try:
            r = self._client.post(self.url, json=body)
            r.raise_for_status()
            data = r.json()
            choice_msg = data["choices"][0]
            content = choice_msg["message"]["content"]
            parsed = json.loads(content)
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as e:
            raise RerankerError(f"llm reranker failed: {e}") from e
        choice = str(parsed.get("choice", "")).strip()
        if choice not in labels:
            # Constrained decoding should prevent this; never accept it.
            return result_from_label_scores(request, {"NO_MATCH": 1.0, choice or "<empty>": 1.0}, self.name,
                                            self.model, explanation="invalid choice returned",
                                            scores_are_probabilities=False,
                                            latency_ms=(time.perf_counter() - t0) * 1000)
        dist = _distribution_from_logprobs(choice_msg.get("logprobs"), content, labels)
        if dist:
            scores, is_prob = dist, True
        else:
            conf = min(max(float(parsed.get("confidence", 1.0)), 0.0), 1.0)
            others = [label for label in labels if label != choice]
            scores = {label: (1 - conf) / len(others) for label in others}
            scores[choice] = conf
            is_prob = False
        return result_from_label_scores(
            request, scores, self.name, self.model, explanation=parsed.get("reason"),
            scores_are_probabilities=is_prob, latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def close(self) -> None:
        self._client.close()


def _distribution_from_logprobs(logprobs: dict | None, content: str, labels: list[str]) -> dict[str, float] | None:
    """Read option probabilities at the first token of the "choice" value."""
    if not logprobs or not logprobs.get("content"):
        return None
    m = re.search(r'"choice"\s*:\s*"', content)
    if not m:
        return None
    target = m.end()
    pos = 0
    for tok in logprobs["content"]:
        text = tok.get("token", "")
        if pos <= target < pos + max(len(text), 1) or (pos >= target and text.strip('"')):
            dist: dict[str, float] = {}
            for alt in tok.get("top_logprobs", []) or [tok]:
                piece = alt.get("token", "").lstrip('"').strip()
                if not piece:
                    continue
                matches = [label for label in labels if label == piece] or [
                    label for label in labels if label.startswith(piece)
                ]
                if len(matches) == 1:
                    dist[matches[0]] = dist.get(matches[0], 0.0) + math.exp(alt.get("logprob", -100.0))
            return dist or None
        pos += len(text)
    return None
