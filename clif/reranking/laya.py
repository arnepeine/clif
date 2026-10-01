"""Laya reranker (https://github.com/NandhaKishorM/laya).

Laya is a non-autoregressive System-1 decision engine: a typed ``choice``
question is scored in a single encoder forward pass and returns a probability
for every supplied option. It cannot generate text, so it cannot invent an
SCTID, which is exactly the property the reranking stage needs.

Two transports, same payload:
- ``LayaHttpReranker``: self-hosted ``laya-serve`` (``POST /v1/systemone``,
  the Jev-compatible wire protocol). Runs on-prem; no text leaves the network.
- ``LayaLocalReranker``: the ``laya`` package in-process, e.g. with a
  checkpoint fine-tuned on SNOMED choices (see ``clif export-laya-training``).

Design decisions that follow from how Laya works:
- Option labels are rendered verbatim as option text, so labels are the SNOMED
  preferred terms (made unique; boolean words such as "Yes"/"No" get their
  semantic tag appended, see Laya's "Honest limits"). Laya's answer keys are
  mapped back to candidates here; nothing else is accepted.
- All options share one ``head_max_len`` token budget. It is sized per request
  from the rendered options so labels are never truncated into each other; if
  Laya still reports collapsed options (``usage.options``), the result is
  flagged and the pipeline returns AMBIGUOUS.
- The state holds only the concept-bearing core text: negation and history are
  split off before reranking, because Laya's documentation reports weaknesses
  on negated input.
- Instructions are short: they compete for the same head budget as options.
"""

from __future__ import annotations

import math
import threading
import time

import httpx

from clif.reranking.base import (
    AMBIGUOUS,
    NO_MATCH,
    ConceptReranker,
    RerankerError,
    RerankRequest,
    RerankResult,
    result_from_label_scores,
)

QUESTION_ID = "snomed_concept"
INSTRUCTIONS = (
    "Which SNOMED CT concept does the clinical text denote? Choose the most specific concept the text fully "
    "supports; never one adding unstated details (acuity, severity, site, laterality, cause). "
    "NO_MATCH if none fits; AMBIGUOUS if the text cannot decide."
)
NO_MATCH_TEXT = "none of the listed concepts represents the clinical text"
AMBIGUOUS_TEXT = "the clinical text does not allow choosing one concept reliably"
_BOOLEAN_WORDS = {"yes", "no", "true", "false", "ja", "nein", "y", "n", "positive", "negative"}
_KNOWN_MODELS = {"english", "multilingual", "typed-decisions"}


def _estimate_tokens(text: str) -> int:
    # Subword tokenizers average ~3-4 chars/token on clinical English; stay pessimistic.
    return math.ceil(len(text) / 3.0) + 1


def build_laya_request(
    request: RerankRequest,
    model: str = "",
    option_synonyms: int = 2,
    head_min: int = 256,
    head_max: int = 1024,
    state_budget: int = 512,
) -> tuple[dict, dict[str, str]]:
    """Return (systemone request body, laya label -> candidate label)."""
    criteria: dict[str, str] = {}
    label_map: dict[str, str] = {}
    used: set[str] = set()
    for c in request.candidates:
        pt = c.preferred_term.strip() or c.concept_id
        label = pt
        if pt.lower() in _BOOLEAN_WORDS or pt.lower() in used or pt in (NO_MATCH, AMBIGUOUS):
            label = f"{pt} ({c.semantic_tag or 'concept'})"
        if label.lower() in used:
            label = f"{pt} [{c.concept_id}]"
        used.add(label.lower())
        desc = c.semantic_tag or "concept"
        # The descriptions retrieval actually matched come first (e.g. "Herzinfarkt" for a German
        # query), so the model compares like with like instead of translating.
        extras = list(dict.fromkeys(t for t in [*c.matched_terms, *c.synonyms] if t.lower() != pt.lower()))
        syns = extras[:option_synonyms]
        if syns:
            desc += "; also: " + ", ".join(syns)
        criteria[label] = desc
        label_map[label] = c.label
    criteria[NO_MATCH] = NO_MATCH_TEXT
    criteria[AMBIGUOUS] = AMBIGUOUS_TEXT
    label_map[NO_MATCH] = NO_MATCH
    label_map[AMBIGUOUS] = AMBIGUOUS

    # Per option: [MASK] + " label: description", capped by Laya at 48 tokens.
    est = _estimate_tokens(INSTRUCTIONS) + 16
    est += sum(min(48, 2 + _estimate_tokens(f"{k}: {v}")) for k, v in criteria.items())
    head = int(min(head_max, max(head_min, math.ceil(est * 1.25 / 64) * 64)))

    state: dict = {"clinical_text": request.core_text}
    if request.expanded_forms:
        state["abbreviation_expansions"] = request.expanded_forms
    if request.context:
        state["context"] = request.context[:4000]
    body: dict = {
        "state": state,
        "questions": {QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
        "head_max_len": head,
        "max_len": head + state_budget,
    }
    chosen_model = model or _model_for_language(request.language)
    if chosen_model:
        body["model"] = chosen_model
    return body, label_map


def _model_for_language(language: str | None) -> str:
    if not language:
        return ""  # let Laya's router detect script/language
    return "english" if language.split("-")[0].lower() == "en" else "multilingual"


def parse_laya_response(data: dict, label_map: dict[str, str]) -> tuple[dict[str, float], dict]:
    """Map Laya's per-label probabilities back to candidate labels."""
    try:
        answer = data["answers"][QUESTION_ID]
        probs = answer["probabilities"]
    except (KeyError, TypeError) as e:
        raise RerankerError(f"unexpected Laya response: {str(data)[:300]}") from e
    scores: dict[str, float] = {}
    for laya_label, p in probs.items():
        # Unknown labels are passed through unmapped so they are rejected and reported.
        scores[label_map.get(laya_label, f"unknown:{laya_label}")] = float(p)
    usage = data.get("usage") or {}
    meta = {
        "laya_choice": answer.get("choice"),
        "laya_answer_confidence": answer.get("answer_confidence"),
        "laya_confidence": answer.get("confidence"),
        "laya_model": data.get("model"),
        "routing": data.get("routing"),
        "input_tokens": usage.get("input_tokens"),
    }
    collapsed = (usage.get("options") or {}).get(QUESTION_ID)
    if collapsed:
        meta["option_collapse"] = collapsed
    return scores, meta


class _LayaBase(ConceptReranker):
    name = "laya"

    def __init__(self, model: str = "", option_synonyms: int = 2, head_min: int = 256, head_max: int = 1024,
                 state_budget: int = 512):
        self.laya_model = model if model in _KNOWN_MODELS else ""
        self.model = model or "laya-router"
        self.option_synonyms = option_synonyms
        self.head_min, self.head_max, self.state_budget = head_min, head_max, state_budget

    def _build(self, request: RerankRequest) -> tuple[dict, dict[str, str]]:
        return build_laya_request(request, self.laya_model, self.option_synonyms, self.head_min, self.head_max,
                                  self.state_budget)

    def _result(self, request: RerankRequest, data: dict, label_map: dict[str, str], t0: float) -> RerankResult:
        scores, meta = parse_laya_response(data, label_map)
        res = result_from_label_scores(
            request, scores, self.name, meta.get("laya_model") or self.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        res.meta = meta
        if meta.get("option_collapse"):
            res.warnings.append("option_collapse")
        return res


class LayaHttpReranker(_LayaBase):
    """Talks to ``laya-serve`` over the Jev-compatible ``/v1/systemone`` protocol."""

    def __init__(self, url: str, api_key: str = "", timeout: float = 20.0, client: httpx.Client | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        if not url:
            raise ValueError("CLIF_LAYA_URL is required for the laya reranker")
        self.url = url.rstrip("/")
        if not self.url.endswith("/v1/systemone"):
            self.url += "/v1/systemone"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

    def rerank(self, request: RerankRequest) -> RerankResult:
        t0 = time.perf_counter()
        body, label_map = self._build(request)
        for attempt in range(3):
            try:
                r = self._client.post(self.url, json=body)
                if r.status_code == 503 and attempt < 2:  # busy; server sends Retry-After
                    time.sleep(min(float(r.headers.get("Retry-After", "0.2") or 0.2), 2.0))
                    continue
                r.raise_for_status()
                return self._result(request, r.json(), label_map, t0)
            except (httpx.HTTPError, ValueError) as e:
                raise RerankerError(f"Laya request failed: {e}") from e
        raise RerankerError("Laya server busy")

    def close(self) -> None:
        self._client.close()


class LayaLocalReranker(_LayaBase):
    """Runs Laya in-process. With ``checkpoint`` set, loads that (fine-tuned) model;
    otherwise uses Laya's Router over the shipped checkpoints."""

    name = "laya-local"

    def __init__(self, checkpoint: str = "", device: str = "", preload: bool = True, **kwargs):
        super().__init__(**kwargs)
        try:
            import laya  # optional dependency: pip install "clif[laya]"
        except ImportError as e:  # pragma: no cover - depends on environment
            raise RerankerError('laya is not installed; pip install "clif[laya]"') from e
        self._lock = threading.Lock()  # one forward pass at a time per model
        if checkpoint:
            self._agent = laya.load(checkpoint, device=device or None)
            self._router = None
            self.model = checkpoint
        else:
            self._agent = None
            self._router = laya.Router(device=device or None, preload=preload)

    def _predict(self, body: dict) -> dict:
        kwargs = {"max_len": body["max_len"], "head_max_len": body["head_max_len"]}
        with self._lock:
            if self._agent is not None:
                return self._agent.predict(body["state"], body["questions"], **kwargs)
            return self._router.predict(body["state"], body["questions"], model=body.get("model"), **kwargs)

    def rerank(self, request: RerankRequest) -> RerankResult:
        t0 = time.perf_counter()
        body, label_map = self._build(request)
        try:
            data = self._predict(body)
        except ValueError as e:
            raise RerankerError(f"Laya rejected the request: {e}") from e
        return self._result(request, data, label_map, t0)

    def rerank_many(self, requests: list[RerankRequest]) -> list[RerankResult]:
        """Batch path for /map/batch: Laya's Router groups compatible requests into
        shared forward passes; a single checkpoint scores them one by one."""
        if self._router is None:
            return [self.rerank(r) for r in requests]
        t0 = time.perf_counter()
        built = [self._build(r) for r in requests]
        items = [
            {"state": b["state"], "questions": b["questions"], "model": b.get("model"), "max_len": b["max_len"],
             "head_max_len": b["head_max_len"]}
            for b, _ in built
        ]
        with self._lock:
            outs = self._router.predict_batch(items)
        return [self._result(r, data, lm, t0) for r, (_, lm), data in zip(requests, built, outs)]
