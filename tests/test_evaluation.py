import json
from pathlib import Path

from clif.config import Settings
from clif.decision import DecisionPolicy
from clif.evaluation.runner import calibrate, compute_metrics, load_dataset, run_cases
from clif.training.laya_export import export_laya_training

DATASET = Path(__file__).resolve().parents[1] / "data" / "eval" / "demo_eval.json"


def test_demo_evaluation(mapper):
    cases = load_dataset(DATASET)
    report = compute_metrics(mapper, run_cases(mapper, cases))
    m = report.metrics
    assert m["candidate_recall@32"] >= 0.95
    assert m["reranking_accuracy_given_present"] >= 0.85
    assert m["top1_accuracy"] >= 0.8
    assert m["wrong_match_rate"] <= 0.05
    assert m["no_match_accuracy"] == 1.0
    assert m["ambiguity_accuracy"] >= 0.75
    assert m["latency_ms"]["p50"] < 100
    for key in ("top5_recall", "semantic_category_accuracy", "context_attribute_accuracy"):
        assert m[key] is not None


def test_calibration_roundtrip(mapper, tmp_path):
    results = run_cases(mapper, load_dataset(DATASET))
    policy, info = calibrate(results)
    assert info["calibration_rows"] > 10
    path = tmp_path / "cal.json"
    policy.save(path)
    loaded = DecisionPolicy.from_settings(Settings(calibration_file=str(path), _env_file=None))
    assert loaded.thresholds == policy.thresholds
    assert loaded.calibrator.fitted == policy.calibrator.fitted
    assert json.loads(path.read_text())["thresholds"]["match"] >= 0.3


def test_laya_training_export(mapper, tmp_path):
    out = tmp_path / "laya.jsonl"
    stats = export_laya_training(mapper, out, n=40, no_match_rate=0.2)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert stats["written"] == len(rows) == 40
    row = rows[0]
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])["snomed_concept"]
    assert gold["label"] in questions["snomed_concept"]["criteria"]
    assert sum(gold["probabilities"].values()) == 1.0
    assert any(r["meta"]["kind"] == "no_match" for r in rows)


def test_negative_examples_and_eval_exclusion(mapper, tmp_path):
    from clif.training.laya_export import export_negative_examples

    out = tmp_path / "neg.jsonl"
    excluded = {"it might rain this afternoon"}
    stats = export_negative_examples(mapper, out, n=30, exclude_texts=excluded)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert stats["written"] == len(rows) > 0
    assert all(json.loads(r["gold"])["snomed_concept"]["label"] == "NO_MATCH" for r in rows)
    assert all(json.loads(r["state"])["clinical_text"] not in excluded for r in rows)
