"""Exports SNOMED constrained-choice training data for fine-tuning Laya.

Laya's base checkpoints are close to chance zero-shot on new decision types;
its documentation recommends fine-tuning. SNOMED itself supplies supervision:
every active synonym (and every translation) is a surface form whose correct
concept is known. For each sampled surface form we run Clif's real retrieval,
render the exact Laya question Clif sends at inference time, and record the
gold label. Three example types:

- positive:     gold concept among the options -> its label
- no_match:     gold concept removed from the options -> NO_MATCH
- specificity:  a parent's term with its children among the options -> the parent
                (arises naturally; counted in the stats)

Output rows follow the LocalLLaMA/typed-decisions layout used by Laya's
fine-tuning notebook: {"id", "workflow", "state", "questions", "gold"} with
JSON-encoded state/questions/gold. Set the notebook's head_max_len to the
value stored in each row (``head_max_len``) so training sees the same budget.

Evaluate on held-out *clinical* phrases, not on SNOMED synonyms: this data
teaches the task format and SNOMED's specificity conventions, and a test set
drawn from the same synonyms would overstate accuracy.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from clif.pipeline import ClifMapper
from clif.reranking.base import NO_MATCH
from clif.reranking.laya import QUESTION_ID, build_laya_request
from clif.text import normalize


def vary_case(text: str, rng: random.Random) -> str:
    """Randomise surface casing with the same distribution for every example type.
    SNOMED terms are capitalised while typed clinical text often is not; a first
    fine-tune learned casing as a shortcut (it answered NO_MATCH for every lower-case
    phrase because all negatives were lower case). Acronyms such as "COPD" are kept."""
    if text.isupper() and len(text) <= 6:
        return text
    base = text.lower()
    x = rng.random()
    if x < 0.45:
        return base
    if x < 0.55:
        return base.upper()
    return base[:1].upper() + base[1:]


# Sampling weight per semantic tag (relative). Everything unlisted gets DEFAULT_TAG_WEIGHT.
TAG_WEIGHTS = {
    "disorder": 1.0, "finding": 1.0, "procedure": 1.0, "situation": 0.8, "observable entity": 0.6,
    "body structure": 0.5, "regime/therapy": 0.6, "morphologic abnormality": 0.4, "substance": 0.35,
    "product": 0.3, "medicinal product": 0.3, "clinical drug": 0.2, "specimen": 0.3, "physical object": 0.3,
    "event": 0.3, "organism": 0.1, "qualifier value": 0.15,
}
DEFAULT_TAG_WEIGHT = 0.1


def export_laya_training(mapper: ClifMapper, out: Path, n: int = 30000, no_match_rate: float = 0.15,
                         languages: list[str] | None = None, seed: int = 13,
                         exclude_texts: set[str] | None = None, shard: tuple[int, int] = (0, 1)) -> dict:
    """``exclude_texts``: phrases of an evaluation set (normalized); they never become
    training queries, so evaluating on that set afterwards stays meaningful.
    ``shard=(i, k)``: process every k-th row starting at i (parallel export)."""
    rng = random.Random(seed)
    exclude = {normalize(t) for t in (exclude_texts or set())}
    conn = mapper.store._conn()
    lang_clause = ""
    params: list = []
    if languages:
        lang_clause = f" AND language IN ({','.join('?' * len(languages))})"
        params = list(languages)
    rows = conn.execute(
        f"SELECT s.concept_id, s.term, s.language, s.is_pt, m.semantic_tag FROM search_terms s "
        f"JOIN concept_meta m ON m.concept_id = s.concept_id WHERE s.type = 'SYN'{lang_clause}", params
    ).fetchall()
    # Weighted random order (Efraimidis-Spirakis): clinical hierarchies are sampled more
    # often than the long tail of organisms, substances and metadata.
    rows.sort(key=lambda r: rng.random() ** (1.0 / TAG_WEIGHTS.get(r[4] or "", DEFAULT_TAG_WEIGHT)), reverse=True)
    rows = [r[:4] for r in rows][shard[0]::shard[1]]
    rng = random.Random(seed * 1000 + shard[0])
    s = mapper.settings
    stats: Counter = Counter()
    written = 0
    with open(out, "w", encoding="utf-8") as fh:
        for cid, term, lang, is_pt in rows:
            if written >= n:
                break
            if normalize(term) in exclude:
                stats["excluded_eval_phrase"] += 1
                continue
            p = mapper.prepare(term, None, lang, None, None)
            if p.analysis.insufficient or p.analysis.unresolved_abbreviations:
                continue
            r = mapper.retriever.retrieve(p.variants, p.languages, s.candidate_count, None, p.category,
                                          p.exclude_tags)
            if not r.candidates:
                continue
            req = mapper._rerank_request(p, None, r)
            ids = [c.concept_id for c in req.candidates]
            kind = "positive"
            if cid in ids and rng.random() < no_match_rate:
                req.candidates = [c for c in req.candidates if c.concept_id != cid]
                kind = "no_match"
            elif cid not in ids:
                kind = "retrieval_miss_no_match"
            if not req.candidates:
                continue
            req.core_text = vary_case(req.core_text, rng)
            body, label_map = build_laya_request(req, "", s.laya_option_synonyms, s.laya_head_max_len_min,
                                                 s.laya_head_max_len_max, s.laya_state_budget)
            inverse = {v: k for k, v in label_map.items()}
            gold_cand = next((c.label for c in req.candidates if c.concept_id == cid), None)
            gold_label = inverse[gold_cand] if gold_cand else NO_MATCH
            if gold_cand:
                rec = mapper.store.concept(cid)
                if rec and any(cid in mapper.store.concept(o).parents for o in ids
                               if o != cid and mapper.store.concept(o)):
                    stats["specificity"] += 1
            criteria = body["questions"][QUESTION_ID]["criteria"]
            probs = {label: (1.0 if label == gold_label else 0.0) for label in criteria}
            fh.write(json.dumps({
                "id": f"snomed-{shard[0]}-{written}",
                "workflow": "snomed_concept_mapping",
                "state": json.dumps(body["state"], ensure_ascii=False),
                "questions": json.dumps(body["questions"], ensure_ascii=False),
                "gold": json.dumps({QUESTION_ID: {"label": gold_label, "probabilities": probs}}, ensure_ascii=False),
                "head_max_len": body["head_max_len"],
                "max_len": body["max_len"],
                "meta": {"concept_id": cid, "language": lang, "is_pt": bool(is_pt), "kind": kind,
                         "snomed_version": mapper.store.snomed_version},
            }, ensure_ascii=False) + "\n")
            stats[kind] += 1
            stats[f"lang:{lang}"] += 1
            written += 1
    return {"written": written, **dict(stats)}


# Non-clinical text: the model must learn that NO_MATCH is right when nothing fits.
_EVERYDAY = [
    "it might rain this afternoon", "ask the daughter to phone the office", "car park entrance on the left",
    "lunch is served at noon", "the printer on ward two is broken", "meeting moved to thursday",
    "bring the forms to reception", "the elevator is out of order", "visiting hours end at eight",
    "coffee machine needs cleaning", "shift change at seven", "the patient's son will pick him up",
    "invoice sent to insurance", "room 12 window does not close", "wifi password changed",
    "taxi ordered for discharge", "remember to sign the consent form", "fire drill next week",
    "keys left at the nurses station", "update the phone list", "christmas party in december",
    "das Wetter ist heute schön", "bitte morgen die Familie anrufen", "Parkhaus Ebene drei",
    "Mittagessen um zwölf", "der Drucker auf Station zwei ist defekt", "Besprechung auf Donnerstag verschoben",
    "Formulare an die Pforte bringen", "der Aufzug ist außer Betrieb", "Besuchszeit endet um acht",
    "Kaffeemaschine reinigen", "Schichtwechsel um sieben", "Rechnung an die Versicherung geschickt",
    "Taxi für die Entlassung bestellt", "Einwilligung unterschreiben lassen", "Schlüssel im Stationszimmer",
]
_NONSENSE_SYLLABLES = ["xy", "zor", "qua", "blim", "trek", "vosh", "pli", "gra", "nuv", "kest", "dro", "fiz"]


def export_negative_examples(mapper: ClifMapper, out: Path, n: int = 1500, seed: int = 29,
                             exclude_texts: set[str] | None = None) -> dict:
    """Appends NO_MATCH examples built from everyday phrases, recombinations of them and
    pseudo-words, each with the candidates Clif's retrieval really returns for them."""
    rng = random.Random(seed)
    s = mapper.settings
    texts: list[str] = list(_EVERYDAY)
    while len(texts) < n * 2:
        kind = rng.random()
        if kind < 0.5:
            a, b = rng.sample(_EVERYDAY, 2)
            texts.append(" ".join(a.split()[: rng.randint(2, 4)] + b.split()[-rng.randint(1, 3):]))
        else:
            texts.append(" ".join("".join(rng.sample(_NONSENSE_SYLLABLES, rng.randint(2, 3)))
                                  for _ in range(rng.randint(1, 3))))
    exclude = {normalize(t) for t in (exclude_texts or set())}
    stats: Counter = Counter()
    written = 0
    with open(out, "a", encoding="utf-8") as fh:
        for text in dict.fromkeys(texts):
            if written >= n:
                break
            if normalize(text) in exclude:
                stats["excluded_eval_phrase"] += 1
                continue
            p = mapper.prepare(text, None, "auto", None, None)
            if p.analysis.insufficient:
                continue
            r = mapper.retriever.retrieve(p.variants, p.languages, s.candidate_count, None, p.category,
                                          p.exclude_tags)
            if not r.candidates:
                continue
            req = mapper._rerank_request(p, None, r)
            req.core_text = vary_case(req.core_text, rng)
            body, _ = build_laya_request(req, "", s.laya_option_synonyms, s.laya_head_max_len_min,
                                         s.laya_head_max_len_max, s.laya_state_budget)
            criteria = body["questions"][QUESTION_ID]["criteria"]
            probs = {label: (1.0 if label == NO_MATCH else 0.0) for label in criteria}
            fh.write(json.dumps({
                "id": f"negative-{written}", "workflow": "snomed_concept_mapping",
                "state": json.dumps(body["state"], ensure_ascii=False),
                "questions": json.dumps(body["questions"], ensure_ascii=False),
                "gold": json.dumps({QUESTION_ID: {"label": NO_MATCH, "probabilities": probs}}, ensure_ascii=False),
                "head_max_len": body["head_max_len"], "max_len": body["max_len"],
                "meta": {"concept_id": None, "kind": "negative_non_clinical",
                         "snomed_version": mapper.store.snomed_version},
            }, ensure_ascii=False) + "\n")
            written += 1
            stats["negative_non_clinical"] += 1
    return {"written": written, **dict(stats)}


def _export_worker(args: tuple) -> dict:
    data_dir, out, n, no_match_rate, languages, seed, exclude, shard = args
    import logging

    from clif.config import Settings

    logging.getLogger().setLevel(logging.WARNING)
    mapper = ClifMapper.from_settings(Settings(data_dir=data_dir))
    return export_laya_training(mapper, Path(out), n, no_match_rate, languages, seed, exclude, shard)


def export_parallel(data_dir: Path, out: Path, n: int, workers: int, no_match_rate: float = 0.15,
                    languages: list[str] | None = None, seed: int = 13,
                    exclude_texts: set[str] | None = None) -> dict:
    """Runs ``export_laya_training`` in ``workers`` processes (one SQLite reader and
    retriever each) over disjoint shards, then concatenates the shard files."""
    import multiprocessing as mp

    per = -(-n // workers)
    parts = [Path(f"{out}.part{i}") for i in range(workers)]
    jobs = [(data_dir, str(parts[i]), per, no_match_rate, languages, seed, exclude_texts, (i, workers))
            for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_export_worker, jobs)
    total: Counter = Counter()
    with open(out, "w", encoding="utf-8") as fh:
        for part, res in zip(parts, results):
            with open(part, encoding="utf-8") as ph:
                for line in ph:
                    fh.write(line)
            part.unlink()
            total.update(res)
    return dict(total)
