"""Confidence calibration and the MATCH / AMBIGUOUS / NO_MATCH decision.

Raw reranker probabilities are not treated as calibrated confidence. The
calibrator (Platt scaling over reranker logit, retrieval score and margin) is
fitted on a validation set by `clif evaluate --calibrate`, which also tunes
the thresholds. Until then an identity calibration is used and every response
reports `calibrated: false`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from clif.config import Settings


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Calibrator:
    a: float = 1.0  # weight of reranker logit
    b: float = 0.0  # weight of retrieval score
    c: float = 0.0  # weight of reranker margin
    bias: float = 0.0
    fitted: bool = False

    def __call__(self, p_rerank: float, retrieval_score: float, margin: float) -> float:
        if not self.fitted:
            return p_rerank
        return _sigmoid(self.a * _logit(p_rerank) + self.b * retrieval_score + self.c * margin + self.bias)

    def fit(self, rows: list[tuple[float, float, float, int]], epochs: int = 3000, lr: float = 0.05,
            l2: float = 1e-3) -> "Calibrator":
        """Logistic regression by gradient descent (no sklearn dependency)."""
        if len(rows) < 10 or len({y for *_, y in rows}) < 2:
            return self
        a, b, c, bias = 1.0, 0.0, 0.0, 0.0
        n = len(rows)
        for _ in range(epochs):
            ga = gb = gc = gbias = 0.0
            for p, r, m, y in rows:
                x = _logit(p)
                err = _sigmoid(a * x + b * r + c * m + bias) - y
                ga += err * x
                gb += err * r
                gc += err * m
                gbias += err
            a -= lr * (ga / n + l2 * a)
            b -= lr * (gb / n + l2 * b)
            c -= lr * (gc / n + l2 * c)
            bias -= lr * gbias / n
        self.a, self.b, self.c, self.bias, self.fitted = a, b, c, bias, True
        return self


@dataclass
class Thresholds:
    match: float = 0.60
    margin: float = 0.20
    no_match: float = 0.30
    plausible: float = 0.15


@dataclass
class DecisionPolicy:
    thresholds: Thresholds = field(default_factory=Thresholds)
    calibrator: Calibrator = field(default_factory=Calibrator)
    source: str = "defaults"

    @classmethod
    def from_settings(cls, settings: Settings) -> "DecisionPolicy":
        th = Thresholds(settings.threshold_match, settings.threshold_margin, settings.threshold_no_match,
                        settings.threshold_plausible)
        policy = cls(th, Calibrator(), "settings")
        if settings.calibration_file and Path(settings.calibration_file).exists():
            data = json.loads(Path(settings.calibration_file).read_text())
            policy.thresholds = Thresholds(**data.get("thresholds", asdict(th)))
            policy.calibrator = Calibrator(**data.get("calibrator", {}))
            policy.source = settings.calibration_file
        return policy

    def save(self, path: Path, extra: dict | None = None) -> None:
        payload = {"thresholds": asdict(self.thresholds), "calibrator": asdict(self.calibrator), **(extra or {})}
        Path(path).write_text(json.dumps(payload, indent=2))


@dataclass
class Decision:
    status: str
    reason: str | None
    top_concept: str | None
    confidence: float
    margin: float
    calibrated_scores: dict[str, float]


def decide(
    policy: DecisionPolicy,
    rerank_scores: dict[str, float],
    no_match: float,
    ambiguous: float,
    retrieval_scores: dict[str, float],
    force_ambiguous_reason: str | None = None,
) -> Decision:
    """Apply calibration and thresholds. `rerank_scores` contain only valid candidates."""
    th = policy.thresholds
    ranked = sorted(rerank_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return Decision("NO_MATCH", "no_valid_candidates", None, 0.0, 0.0, {})
    def gap(i: int, p: float) -> float:
        """Margin feature: distance to the best competing candidate."""
        if len(ranked) == 1:
            return p
        best_other = ranked[1][1] if i == 0 else ranked[0][1]
        return p - best_other

    cal = {cid: policy.calibrator(p, retrieval_scores.get(cid, 0.0), gap(i, p)) for i, (cid, p) in enumerate(ranked)}
    cranked = sorted(cal.items(), key=lambda kv: (-kv[1], kv[0]))
    top_id, c1 = cranked[0]
    c2 = cranked[1][1] if len(cranked) > 1 else 0.0
    margin = c1 - c2

    if force_ambiguous_reason:
        return Decision("AMBIGUOUS", force_ambiguous_reason, top_id, c1, margin, cal)
    p_top = ranked[0][1]
    if no_match > p_top and no_match >= ambiguous:
        return Decision("NO_MATCH", "reranker_selected_no_match", top_id, c1, margin, cal)
    if ambiguous > p_top:
        return Decision("AMBIGUOUS", "reranker_selected_ambiguous", top_id, c1, margin, cal)
    if c1 >= th.match and margin >= th.margin:
        return Decision("MATCH", None, top_id, c1, margin, cal)
    if c1 < th.no_match:
        return Decision("NO_MATCH", f"top_confidence_below_no_match_threshold({c1:.2f}<{th.no_match})",
                        top_id, c1, margin, cal)
    if c2 >= th.plausible and margin < th.margin:
        return Decision("AMBIGUOUS", f"close_competitors(margin {margin:.2f}<{th.margin})", top_id, c1, margin, cal)
    return Decision("AMBIGUOUS", f"below_match_threshold({c1:.2f}<{th.match})", top_id, c1, margin, cal)
