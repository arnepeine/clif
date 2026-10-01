"""Synthetic full-scale benchmark (no SNOMED license needed).

Generates an RF2 package with SNOMED International-like volume (default
350k concepts, ~4 descriptions each, 3-level hierarchy) from a vocabulary of
clinical word parts, imports it, builds the index and measures retrieval and
end-to-end latency. Terms are synthetic, so only timings and memory are
meaningful, not accuracy.

    python scripts/benchmark_scale.py --concepts 350000 --workdir /tmp/clif-scale
"""

from __future__ import annotations

import argparse
import random
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clif.config import GB_ENGLISH_REFSET, US_ENGLISH_REFSET, Settings  # noqa: E402
from clif.indexing.build import build_index  # noqa: E402
from clif.models.schemas import MapRequest  # noqa: E402
from clif.pipeline import ClifMapper  # noqa: E402
from clif.reranking import HeuristicReranker  # noqa: E402
from clif.terminology.rf2 import FSN_TYPE, INFERRED, IS_A, PREFERRED, SYNONYM_TYPE, import_packages  # noqa: E402
from clif.terminology.sctid import make_sctid  # noqa: E402

ROOTS = "cardi hepat nephr gastr pulmon neur derm arthr oste my encephal col append cholecyst pancreat thyr " \
        "aden angi phleb lymph splen cyst prostat oophor salping hyster mast rhin laryng trache bronch pleur " \
        "periton ot ophthalm retin kerat blephar gingiv gloss stomat esophag duoden ile jejun proct".split()
SUFFIXES = "itis osis ectomy otomy algia oma pathy plasty scopy megaly sclerosis stenosis".split()
MODS = "acute chronic left right bilateral severe mild recurrent congenital primary secondary partial total " \
       "infective traumatic idiopathic benign malignant upper lower anterior posterior".split()
TAGS = ["disorder", "finding", "procedure", "body structure"]


def make_term(rng: random.Random) -> str:
    words = [rng.choice(ROOTS) + rng.choice(SUFFIXES)]
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        words.insert(0, rng.choice(MODS))
    if rng.random() < 0.3:
        words.append("of " + rng.choice(ROOTS) + "a")
    return " ".join(words).capitalize()


def write_package(root: Path, n: int, seed: int = 7) -> Path:
    rng = random.Random(seed)
    pkg = root / "SnomedCT_ScaleRF2_PRODUCTION_20250101T120000Z" / "Snapshot"
    (pkg / "Terminology").mkdir(parents=True, exist_ok=True)
    (pkg / "Refset" / "Language").mkdir(parents=True, exist_ok=True)
    m = "900000000000207008"
    ids = [make_sctid(1000000 + i) for i in range(n)]
    with open(pkg / "Terminology/sct2_Concept_Snapshot_INT_20250101.txt", "w") as c, \
         open(pkg / "Terminology/sct2_Description_Snapshot-en_INT_20250101.txt", "w") as d, \
         open(pkg / "Terminology/sct2_Relationship_Snapshot_INT_20250101.txt", "w") as r, \
         open(pkg / "Refset/Language/der2_cRefset_LanguageSnapshot-en_INT_20250101.txt", "w") as lang:
        c.write("id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n")
        d.write("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n")
        r.write("id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\t"
                "characteristicTypeId\tmodifierId\n")
        lang.write("id\teffectiveTime\tactive\tmoduleId\trefsetId\treferencedComponentId\tacceptabilityId\n")
        di = ri = li = 0
        for i, cid in enumerate(ids):
            c.write(f"{cid}\t20250101\t{0 if i % 50 == 49 else 1}\t{m}\t900000000000074008\n")
            pt = make_term(rng)
            tag = TAGS[i % 4]
            terms = [(FSN_TYPE, f"{pt} ({tag})", True)] + [(SYNONYM_TYPE, pt, True)] + [
                (SYNONYM_TYPE, make_term(rng), False) for _ in range(rng.choice([1, 2, 3]))]
            for typ, term, pref in terms:
                di += 1
                did = make_sctid(50000000 + di, "10")
                d.write(f"{did}\t20250101\t1\t{m}\t{cid}\ten\t{typ}\t{term}\t900000000000448009\n")
                for refset in (US_ENGLISH_REFSET, GB_ENGLISH_REFSET):
                    li += 1
                    lang.write(f"{li:032x}\t20250101\t1\t{m}\t{refset}\t{did}\t"
                               f"{PREFERRED if pref else '900000000000549004'}\n")
            if i >= 20:
                ri += 1
                parent = ids[rng.randrange(0, max(1, i // 10))]
                r.write(f"{make_sctid(80000000 + ri, '20')}\t20250101\t1\t{m}\t{cid}\t{parent}\t0\t{IS_A}\t"
                        f"{INFERRED}\t900000000000451002\n")
    return pkg.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=350_000)
    ap.add_argument("--workdir", default="/tmp/clif-scale")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--embedder", default="hashing", choices=["none", "hashing"])
    a = ap.parse_args()
    work = Path(a.workdir)
    s = Settings(data_dir=work / "data", embedder=a.embedder, _env_file=None)
    t = time.time()
    pkg = write_package(work / "rf2", a.concepts)
    print(f"generate: {time.time() - t:.1f}s")
    t = time.time()
    st = import_packages([pkg], s.db_path, reset=True)
    print(f"import:   {time.time() - t:.1f}s  rows={st.rows}")
    t = time.time()
    b = build_index(s)
    print(f"index:    {time.time() - t:.1f}s  search_terms={b.search_terms} vectors={b.vectors}")
    m = ClifMapper.from_settings(s, reranker=HeuristicReranker())
    rng = random.Random(99)
    queries = [make_term(rng).lower() for _ in range(a.queries)]
    typos = [q[:3] + q[4:] for q in queries[: a.queries // 3]]  # deletion typo -> fuzzy path
    for label, qs in (("exact/phrase", queries), ("typo", typos)):
        ret, tot = [], []
        for q in qs:
            p = m.prepare(q, None, "en", None, None)
            t0 = time.perf_counter()
            m.retriever._cache.data.clear()
            m._retrieve(p)
            ret.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            m.retriever._cache.data.clear()
            m.map(MapRequest(text=q, language="en"))
            tot.append((time.perf_counter() - t0) * 1000)
        q95 = lambda xs: statistics.quantiles(xs, n=20)[18]  # noqa: E731
        print(f"{label:13s} retrieval p50={statistics.median(ret):6.1f}ms p95={q95(ret):6.1f}ms | "
              f"map p50={statistics.median(tot):6.1f}ms p95={q95(tot):6.1f}ms")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak RSS: {rss / (1024 * 1024 if sys.platform == 'darwin' else 1024):.0f} MB")


if __name__ == "__main__":
    main()
