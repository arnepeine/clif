"""Command line interface.

    clif demo-release ./data/rf2
    clif import-rf2 /data/SnomedCT_InternationalRF2_PRODUCTION_20250101T120000Z [more packages...]
    clif build-index
    clif serve --host 0.0.0.0 --port 8000
    clif map "acute appendicitis" [--context ...] [--debug]
    clif candidates "Herzinfarkt" --language de
    clif evaluate data/eval/demo_eval.json [--calibrate calibration.json] [--report report.json]
    clif export-laya-training --out laya_snomed.jsonl --n 30000
    clif train-laya --data laya_snomed.jsonl --out models/laya-snomed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from clif.config import get_settings


def _mapper():
    from clif.pipeline import ClifMapper

    return ClifMapper.from_settings(get_settings())


def cmd_demo_release(a) -> int:
    from clif.devtools.demo_release import write_demo_release

    print(write_demo_release(Path(a.target)))
    return 0


def cmd_import(a) -> int:
    from clif.terminology.rf2 import import_packages

    s = get_settings()
    stats = import_packages([Path(p) for p in a.packages], s.db_path, a.release_type, reset=not a.append,
                            validate_ids=a.validate_ids)
    print(json.dumps({"db": str(s.db_path), "rows": stats.rows, "invalid_ids": stats.invalid_ids,
                      "snomed_version": stats.snomed_version, "seconds": round(stats.seconds, 1)}, indent=2))
    return 0


def cmd_build(a) -> int:
    from clif.indexing.build import build_index

    stats = build_index(get_settings())
    print(json.dumps(stats.__dict__, indent=2, default=str))
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    uvicorn.run("clif.api:app", host=a.host, port=a.port, workers=a.workers, log_level="info")
    return 0


def cmd_map(a) -> int:
    from clif.models.schemas import MapRequest

    res = _mapper().map(MapRequest(text=a.text, context=a.context, language=a.language, debug=a.debug,
                                   allowed_hierarchies=a.hierarchy or None, ecl=a.ecl))
    print(res.model_dump_json(indent=2, exclude_none=not a.debug))
    return 0


def cmd_candidates(a) -> int:
    from clif.models.schemas import CandidatesRequest

    res = _mapper().candidates(CandidatesRequest(text=a.text, context=a.context, language=a.language,
                                                 limit=a.limit))
    if a.json:
        print(res.model_dump_json(indent=2))
    else:
        print(f"core='{res.core_text}' languages={res.searched_languages} ({res.elapsed_ms} ms)")
        for c in res.candidates:
            m = c.matched_descriptions[0] if c.matched_descriptions else {}
            print(f"{c.rank:3d} {c.retrieval_score:.3f} {c.concept_id:>18} {c.preferred_term[:40]:40s} "
                  f"[{c.semantic_tag}] via '{m.get('term', '')}' ({m.get('language', '')})")
    return 0


def cmd_evaluate(a) -> int:
    from clif.evaluation.runner import calibrate, compute_metrics, load_dataset, run_cases

    mapper = _mapper()
    cases = load_dataset(Path(a.dataset))
    results = run_cases(mapper, cases)
    report = compute_metrics(mapper, results)
    out = report.to_dict()
    if a.calibrate:
        policy, info = calibrate(results, a.wrong_match_penalty)
        policy.save(Path(a.calibrate), {"fitted_on": a.dataset, "n": len(cases), "info": info,
                                        "snomed_version": mapper.store.snomed_version,
                                        "reranker": mapper.reranker.name})
        out["calibration"] = {"file": a.calibrate, **info, "thresholds": policy.thresholds.__dict__}
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if a.report:
        Path(a.report).write_text(text)
    print(text if a.verbose else json.dumps({"n": out["n"], "metrics": out["metrics"],
                                             "failures": len(out["failures"]),
                                             **({"calibration": out["calibration"]} if a.calibrate else {})},
                                            indent=2))
    return 0


def cmd_export_laya(a) -> int:
    from clif.training.laya_export import export_laya_training

    exclude: set[str] = set()
    for path in a.exclude_eval or []:
        from clif.evaluation.runner import load_dataset

        exclude |= {c["text"] for c in load_dataset(Path(path))}
    mapper = _mapper()
    n = 10**9 if a.all else a.n
    if a.workers > 1:
        from clif.training.laya_export import export_parallel

        stats = export_parallel(get_settings().data_dir, Path(a.out), n, a.workers, a.no_match_rate,
                                a.language or None, a.seed, exclude)
    else:
        stats = export_laya_training(mapper, Path(a.out), n, a.no_match_rate, a.language or None, a.seed,
                                     exclude)
    if a.negatives:
        from clif.training.laya_export import export_negative_examples

        stats["negatives"] = export_negative_examples(mapper, Path(a.out), a.negatives, exclude_texts=exclude)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train_laya(a) -> int:
    from clif.training.train_laya import TrainConfig, train

    info = train(TrainConfig(data=Path(a.data), out=Path(a.out), base=a.base, subfolder=a.subfolder,
                             epochs=a.epochs, micro_batch=a.micro_batch, grad_accum=a.grad_accum,
                             max_items=a.max_items, device=a.device, save_every=a.save_every,
                             holdout_max=a.holdout_max, resume=a.resume))
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clif", description="Clinical free text to SNOMED CT")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo-release", help="write the small demo RF2 package (tests/quickstart)")
    s.add_argument("target", nargs="?", default="./data/rf2")
    s.set_defaults(fn=cmd_demo_release)

    s = sub.add_parser("import-rf2", help="import one or more RF2 packages (dirs or zips)")
    s.add_argument("packages", nargs="+")
    s.add_argument("--release-type", default="snapshot", choices=["snapshot", "full", "delta"])
    s.add_argument("--append", action="store_true", help="upsert into the existing DB instead of recreating it")
    s.add_argument("--validate-ids", action="store_true", help="count SCTIDs with invalid Verhoeff check digit")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("build-index", help="build search tables, FTS indexes and vectors")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_serve)

    for name, fn in (("map", cmd_map), ("candidates", cmd_candidates)):
        s = sub.add_parser(name)
        s.add_argument("text")
        s.add_argument("--context")
        s.add_argument("--language", default="auto")
        if name == "map":
            s.add_argument("--debug", action="store_true")
            s.add_argument("--hierarchy", action="append")
            s.add_argument("--ecl")
        else:
            s.add_argument("--limit", type=int, default=32)
            s.add_argument("--json", action="store_true")
        s.set_defaults(fn=fn)

    s = sub.add_parser("evaluate", help="evaluate on a labelled dataset; optionally calibrate")
    s.add_argument("dataset")
    s.add_argument("--calibrate", metavar="OUT_JSON", help="fit calibrator + thresholds and write them here")
    s.add_argument("--wrong-match-penalty", type=float, default=4.0)
    s.add_argument("--report", metavar="REPORT_JSON")
    s.add_argument("-v", "--verbose", action="store_true", help="print failures too")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("export-laya-training", help="export SNOMED choice data for fine-tuning Laya")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=30000)
    s.add_argument("--no-match-rate", type=float, default=0.15)
    s.add_argument("--language", action="append")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--exclude-eval", action="append", metavar="DATASET",
                   help="evaluation set whose phrases must not appear as training queries")
    s.add_argument("--all", action="store_true", help="every searchable synonym (ignores --n)")
    s.add_argument("--workers", type=int, default=1, help="parallel export processes")
    s.add_argument("--negatives", type=int, default=0,
                   help="append this many non-clinical NO_MATCH examples (everyday phrases, pseudo-words)")
    s.set_defaults(fn=cmd_export_laya)

    s = sub.add_parser("train-laya", help="fine-tune a Laya checkpoint on exported SNOMED choice data")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--base", default="convaiinnovations/laya", help="HF repo id or local checkpoint dir")
    s.add_argument("--subfolder", default="multilingual", help="checkpoint inside the repo ('' = English root)")
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--micro-batch", type=int, default=8)
    s.add_argument("--grad-accum", type=int, default=4)
    s.add_argument("--max-items", type=int, default=0)
    s.add_argument("--device", default="", help="mps | cuda | cpu (default: best available)")
    s.add_argument("--save-every", type=int, default=5000, help="optimizer steps between rolling checkpoints")
    s.add_argument("--holdout-max", type=int, default=3000)
    s.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint_latest")
    s.set_defaults(fn=cmd_train_laya)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
