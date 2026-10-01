"""Evaluation, calibration and threshold tuning.

Dataset (JSON list or JSONL), one object per case:
    {"text": "acute appendicitis", "expected_concept_id": "85189001"}
    {"text": "the weather is nice", "expected_status": "NO_MATCH"}
    {"text": "MS", "expected_status": "AMBIGUOUS"}
Optional fields: context, language, allowed_hierarchies, ecl, expected_category,
expected_assertion, expected_temporality, expected_certainty.
`expected_status` defaults to MATCH when `expected_concept_id` is given, else NO_MATCH.

The key split: candidate recall (did retrieval find the right concept?) versus
reranking accuracy given the concept was offered (did the model pick it?).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from clif.decision import Calibrator, DecisionPolicy, Thresholds, decide
from clif.models.schemas import MapRequest
from clif.pipeline import ClifMapper
from clif.retrieval.category import category_of_tag


def load_dataset(path: Path) -> list[dict]:
    text = Path(path).read_text("utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    for i, row in enumerate(data):
        if "text" not in row:
            raise ValueError(f"case {i}: missing 'text'")
        row.setdefault("expected_status", "MATCH" if row.get("expected_concept_id") else "NO_MATCH")
        if row.get("expected_concept_id") is not None:
            row["expected_concept_id"] = str(row["expected_concept_id"])
    return data


@dataclass
class CaseResult:
    case: dict
    status: str
    concept_id: str | None
    reason: str | None
    latency_ms: float
    retrieval_ids: list[str]
    rerank_ids: list[str]  # candidates offered to the reranker
    rerank_scores: dict[str, float]
    rerank_no_match: float
    rerank_ambiguous: float
    retrieval_scores: dict[str, float]
    ranked: list[str]  # reranker order
    predicted_category: str
    assertion: str
    temporality: str
    certainty: str
    forced_reason: str | None = None
    failure: str | None = None


@dataclass
class EvalReport:
    n: int
    metrics: dict
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n": self.n, "metrics": self.metrics, "failures": self.failures}


def _pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 2) if values else 0.0


def run_cases(mapper: ClifMapper, cases: list[dict]) -> list[CaseResult]:
    out: list[CaseResult] = []
    for case in cases:
        req = MapRequest(
            text=case["text"], context=case.get("context"), language=case.get("language", "auto"),
            allowed_hierarchies=case.get("allowed_hierarchies"), ecl=case.get("ecl"), top_k=32, debug=True,
        )
        t0 = time.perf_counter()
        res = mapper.map(req)
        latency = (time.perf_counter() - t0) * 1000
        d = res.debug or {}
        retr = [c["concept_id"] for c in (d.get("retrieval") or {}).get("candidates", [])]
        rr = d.get("reranker") or {}
        probs: dict[str, float] = rr.get("probabilities", {})
        ranked = [cid for cid, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
        retr_scores = {c["concept_id"]: c["retrieval_score"] for c in (d.get("retrieval") or {}).get("candidates", [])}
        reason = res.reason or ""
        forced = reason if reason.startswith(("ambiguous_abbreviation", "reranker_option_collapse")) else None
        out.append(CaseResult(
            case=case, status=res.status, concept_id=res.concept.concept_id if res.concept else None,
            reason=res.reason, latency_ms=latency, retrieval_ids=retr, rerank_ids=list(probs.keys()),
            rerank_scores=probs, rerank_no_match=rr.get("no_match", 0.0), rerank_ambiguous=rr.get("ambiguous", 0.0),
            retrieval_scores=retr_scores, ranked=ranked,
            predicted_category=(d.get("semantic_category") or {}).get("top", "other"),
            assertion=res.assertion, temporality=res.temporality, certainty=res.certainty, forced_reason=forced,
        ))
    return out


def compute_metrics(mapper: ClifMapper, results: list[CaseResult], candidate_k: int | None = None) -> EvalReport:
    k = candidate_k or mapper.settings.candidate_count
    match_cases = [r for r in results if r.case["expected_status"] == "MATCH"]
    top1 = top5 = recall_k = recall_rr = present = rerank_ok = mapped_ok_present = wrong = 0
    failures: list[dict] = []
    for r in match_cases:
        exp = r.case["expected_concept_id"]
        correct = r.status == "MATCH" and r.concept_id == exp
        top1 += correct
        top5 += exp == r.concept_id or exp in r.ranked[:5]
        in_k = exp in r.retrieval_ids[:k]
        recall_k += in_k
        in_rr = exp in r.rerank_ids
        recall_rr += in_rr
        if in_rr:
            present += 1
            rerank_ok += bool(r.ranked) and r.ranked[0] == exp
            mapped_ok_present += correct
        if r.status == "MATCH" and r.concept_id != exp:
            wrong += 1
        if not correct:
            r.failure = ("retrieval_miss" if not in_k else "not_offered_to_reranker" if not in_rr
                         else "reranker_wrong_choice" if r.ranked and r.ranked[0] != exp
                         else f"decision_{r.status.lower()}")
            failures.append({"text": r.case["text"], "expected": exp, "got_status": r.status, "got": r.concept_id,
                             "failure": r.failure, "reason": r.reason})
    by_status: dict[str, list[CaseResult]] = {}
    for r in results:
        by_status.setdefault(r.case["expected_status"], []).append(r)
    for r in results:
        if r.case["expected_status"] != "MATCH" and r.status != r.case["expected_status"]:
            if r.status == "MATCH":
                wrong += 1
            failures.append({"text": r.case["text"], "expected_status": r.case["expected_status"],
                             "got_status": r.status, "got": r.concept_id, "failure": "wrong_status",
                             "reason": r.reason})

    def acc(status: str) -> float | None:
        rs = by_status.get(status, [])
        return round(sum(r.status == status for r in rs) / len(rs), 4) if rs else None

    cat_total = cat_ok = 0
    for r in results:
        exp_cat = r.case.get("expected_category")
        if exp_cat is None and r.case.get("expected_concept_id"):
            rec = mapper.store.concept(r.case["expected_concept_id"])
            exp_cat = category_of_tag(rec.semantic_tag) if rec else None
        if exp_cat:
            cat_total += 1
            cat_ok += r.predicted_category == exp_cat

    ctx_total = ctx_ok = 0
    for r in results:
        for fld in ("assertion", "temporality", "certainty"):
            if f"expected_{fld}" in r.case:
                ctx_total += 1
                ctx_ok += getattr(r, fld) == r.case[f"expected_{fld}"]

    lat = [r.latency_ms for r in results]
    n_match = len(match_cases) or 1
    metrics = {
        "top1_accuracy": round(top1 / n_match, 4),
        "top5_recall": round(top5 / n_match, 4),
        f"candidate_recall@{k}": round(recall_k / n_match, 4),
        f"candidate_recall@{mapper.settings.rerank_candidates}_offered_to_reranker": round(recall_rr / n_match, 4),
        "reranking_accuracy_given_present": round(rerank_ok / present, 4) if present else None,
        "mapping_accuracy_given_present": round(mapped_ok_present / present, 4) if present else None,
        "no_match_accuracy": acc("NO_MATCH"),
        "ambiguity_accuracy": acc("AMBIGUOUS"),
        "insufficient_information_accuracy": acc("INSUFFICIENT_INFORMATION"),
        "wrong_match_rate": round(wrong / len(results), 4) if results else 0.0,
        "semantic_category_accuracy": round(cat_ok / cat_total, 4) if cat_total else None,
        "context_attribute_accuracy": round(ctx_ok / ctx_total, 4) if ctx_total else None,
        "latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95), "p99": _pct(lat, 99)},
        "counts": {s: len(v) for s, v in by_status.items()},
        "reranker": mapper.reranker.name,
        "snomed_version": mapper.store.snomed_version,
    }
    return EvalReport(n=len(results), metrics=metrics, failures=failures)


def calibrate(results: list[CaseResult], wrong_match_penalty: float = 4.0) -> tuple[DecisionPolicy, dict]:
    """Fit the confidence calibrator, then grid-search thresholds that maximise
    correct MATCHes minus `wrong_match_penalty` x wrong MATCHes."""
    rows = []
    for r in results:
        if not r.ranked:
            continue
        top = r.ranked[0]
        p1 = r.rerank_scores[top]
        p2 = r.rerank_scores[r.ranked[1]] if len(r.ranked) > 1 else 0.0
        label = int(r.case["expected_status"] == "MATCH" and top == r.case.get("expected_concept_id"))
        rows.append((p1, r.retrieval_scores.get(top, 0.0), p1 - p2, label))
    cal = Calibrator().fit(rows)

    def utility(th: Thresholds) -> tuple[float, int, int]:
        pol = DecisionPolicy(th, cal)
        good = bad = 0
        for r in results:
            if not r.rerank_scores:
                continue
            valid = {cid: p for cid, p in r.rerank_scores.items()}
            dec = decide(pol, valid, r.rerank_no_match, r.rerank_ambiguous, r.retrieval_scores, r.forced_reason)
            if dec.status != "MATCH":
                continue
            if r.case["expected_status"] == "MATCH" and dec.top_concept == r.case.get("expected_concept_id"):
                good += 1
            else:
                bad += 1
        return good - wrong_match_penalty * bad, good, bad

    best = None
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.0, 0.51, 0.05), 2)
    for tm, mg in product(grid, margins):
        th = Thresholds(match=float(tm), margin=float(mg), no_match=min(0.3, float(tm)), plausible=0.15)
        u, g, b = utility(th)
        # Ties go to the stricter setting.
        if best is None or u > best[0] or (u == best[0] and (tm, mg) > (best[1].match, best[1].margin)):
            best = (u, th, g, b)
    assert best is not None
    policy = DecisionPolicy(best[1], cal, "calibration")
    info = {"utility": best[0], "correct_matches": best[2], "wrong_matches": best[3],
            "wrong_match_penalty": wrong_match_penalty, "calibration_rows": len(rows),
            "calibrator_fitted": cal.fitted}
    return policy, info
