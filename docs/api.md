# API reference

Start the server with `uvicorn clif.api:app --host 0.0.0.0 --port 8000`, or `clif serve`. Interactive OpenAPI docs are at `/docs`.

All examples below are real responses from the bundled demo release with the rule-based judge.

| endpoint | purpose |
|---|---|
| `POST /map` | map one phrase |
| `POST /map/batch` | map many phrases in one call |
| `POST /candidates` | retrieval only, without the judge (debugging) |
| `GET /concepts/{id}` | look up one concept |
| `GET /health` | readiness, SNOMED version, active judge |

Errors:

| code | when |
|---|---|
| `400` | invalid or unsupported ECL, unknown hierarchy name, text longer than the configured limit |
| `413` | batch larger than `CLIF_MAX_BATCH_SIZE` |
| `422` | malformed request body |
| `503` | index not built yet, or the judge backend is unreachable (and no fallback is configured) |

## POST /map

### Request

| field | type | default | |
|---|---|---|---|
| `text` | string | required | the phrase to map |
| `context` | string | `null` | surrounding text; used to resolve abbreviations and passed to the judge, never searched itself |
| `language` | string | `"auto"` | source language (`en`, `de`, `nl`, `fr`, ...) or `auto` |
| `display_language` | string | `CLIF_DEFAULT_DISPLAY_LANGUAGE` | language of the returned preferred terms |
| `top_k` | int 0..32 | `5` | number of alternatives |
| `allowed_hierarchies` | list | `null` | e.g. `["procedure"]` or top-level SCTIDs |
| `ecl` | string | `null` | ECL restriction, e.g. `"<< 64572001 \|Disease\|"` |
| `debug` | bool | `false` | include the full decision trace |

```bash
curl -s localhost:8000/map -H 'content-type: application/json' \
  -d '{"text": "patient denies chest pain", "top_k": 2}'
```

### Response

```json
{
  "status": "MATCH",
  "input": "patient denies chest pain",
  "context_supplied": false,
  "concept": {
    "concept_id": "29857009",
    "preferred_term": "Chest pain",
    "fsn": "Chest pain (finding)",
    "semantic_tag": "finding",
    "matched_description": {"description_id": "99980046107", "term": "Chest pain", "language": "en", "type": "SYN"}
  },
  "confidence": 0.9998,
  "alternatives": [
    {"concept_id": "22253000", "preferred_term": "Pain", "semantic_tag": "finding",
     "confidence": 0.0001, "reranker_score": 0.0001, "retrieval_score": 0.5344, "retrieval_rank": 2},
    {"concept_id": "51185008", "preferred_term": "Thoracic structure", "semantic_tag": "body structure",
     "confidence": 0.0001, "reranker_score": 0.0001, "retrieval_score": 0.4909, "retrieval_rank": 3}
  ],
  "assertion": "NEGATED",
  "temporality": "CURRENT",
  "certainty": "CERTAIN",
  "experiencer": "PATIENT",
  "reason": null,
  "warnings": [],
  "source_language": "en",
  "matched_description_language": "en",
  "display_language": "en",
  "semantic_category_hint": null,
  "snomed_version": "20250101",
  "snomed_edition": "SnomedCT_ClifDemoRF2_PRODUCTION_20250101T120000Z",
  "reranker": {"backend": "heuristic", "model": "bidirectional-coverage-v1", "fallback_used": false},
  "calibrated": false,
  "clif_version": "0.1.0",
  "debug": null
}
```

| field | meaning |
|---|---|
| `status` | `MATCH`, `AMBIGUOUS`, `NO_MATCH` or `INSUFFICIENT_INFORMATION` |
| `concept` | set only for `MATCH`. `matched_description` is the description the text matched through, which may be a synonym or a translation |
| `confidence` | calibrated confidence of `concept`; `0.0` when there is no concept |
| `alternatives` | other candidates with their own confidences, also for `AMBIGUOUS` and `NO_MATCH`, so a human can review them |
| `assertion`, `temporality`, `certainty`, `experiencer` | context separated from the concept |
| `reason` | why the status isn't `MATCH`, e.g. `reranker_selected_no_match`, `close_competitors(margin 0.08<0.2)`, `ambiguous_abbreviation: ...` |
| `source_language` | language of the input (detected or given) |
| `matched_description_language` | language of the description that matched |
| `display_language` | language of the returned preferred terms |
| `snomed_version`, `snomed_edition` | the release used; store these with the mapping for reproducibility |
| `reranker` | judge backend and model; `fallback_used` is true if the primary judge failed |
| `calibrated` | false until thresholds were fitted with `clif evaluate --calibrate` |

### An ambiguous abbreviation

```json
{
  "status": "AMBIGUOUS",
  "concept": null,
  "confidence": 0.0,
  "alternatives": [
    {"concept_id": "24700007", "preferred_term": "Multiple sclerosis", "confidence": 0.3328, "...": "..."},
    {"concept_id": "79619009", "preferred_term": "Mitral valve stenosis", "confidence": 0.2993, "...": "..."}
  ],
  "reason": "ambiguous_abbreviation: MS = multiple sclerosis (0.60) | mitral stenosis (0.40)"
}
```

The same request with `"context": "MRI shows periventricular demyelinating lesions, relapsing course"` returns `MATCH` for *Multiple sclerosis*.

### Debug output

With `"debug": true` the response carries a `debug` object:

| key | content |
|---|---|
| `normalized_query` | every query variant with its normalized form, source and weight |
| `context_analysis` | core text, cue spans, abbreviation posteriors with the context cues that drove them |
| `language` | detected and searched languages; whether the source language was inferred from the matched description |
| `semantic_category` | category distribution and the evidence behind it |
| `constraint` | the effective hierarchy / ECL restriction |
| `retrieval` | all candidates with channel scores and matched descriptions, filtered concepts with reasons, timings |
| `reranker` | probability per offered concept, `no_match`, `ambiguous`, rejected labels, backend metadata (e.g. Laya's own `answer_confidence`) |
| `decision` | status, reason, confidence, margin, thresholds and where they came from |
| `selected` | retrieval rank and scores of the chosen concept |
| `validation` | the terminology checks and their outcome |
| `timing_ms` | retrieval, judge and total time |

## POST /map/batch

```json
{"items": [{"text": "appendectomy"}, {"text": "MS"}, {"text": "kein Fieber", "language": "de"}]}
```

Returns `{"results": [...], "elapsed_ms": ...}` in input order. Each result is a `/map` response or `{"status": "ERROR", "input": ..., "error": ...}`, so one bad item doesn't fail the batch.

Batching embeds all queries in one call and shares the retrieval cache. With `laya-local` it also scores the items in batched forward passes; other judges run with bounded concurrency (`CLIF_RERANKER_CONCURRENCY`).

## POST /candidates

Retrieval without the judge. This is the first thing to look at when a mapping is wrong: if the right concept isn't in this list, no judge can choose it.

```bash
curl -s localhost:8000/candidates -H 'content-type: application/json' \
  -d '{"text": "Herzinfarkt", "language": "de", "limit": 32}'
```

```json
{
  "input": "Herzinfarkt",
  "core_text": "Herzinfarkt",
  "queries": [{"text": "Herzinfarkt", "normalized": "herzinfarkt", "folded": "herzinfarkt", "source": "core", "weight": 1.0}],
  "source_language": "de",
  "searched_languages": ["de", "en"],
  "candidates": [
    {
      "rank": 1,
      "concept_id": "22298006",
      "preferred_term": "Myocardial infarction",
      "fsn": "Myocardial infarction (disorder)",
      "semantic_tag": "disorder",
      "retrieval_score": 0.98,
      "lexical_score": 1.0,
      "semantic_score": 1.0,
      "exact_or_synonym_bonus": 1.0,
      "terminology_prior": 0.8,
      "category_factor": 1.0,
      "matched_descriptions": [
        {"description_id": "99980119103", "term": "Herzinfarkt", "language": "de", "type": "SYN",
         "channels": ["bm25", "exact", "prefix", "vector"], "lexical": 1.0, "semantic": 1.0, "query": "Herzinfarkt"}
      ]
    }
  ],
  "filtered": [],
  "descriptions_considered": 187,
  "concepts_considered": 91,
  "channel_counts": {"exact": 1, "prefix": 1, "bm25": 1, "vector": 187},
  "vector_enabled": true,
  "snomed_version": "20250101",
  "elapsed_ms": 2.95
}
```

`limit` accepts 1 to 200. The response also includes `context_analysis` and `semantic_category` as in `/map` debug output.

## GET /concepts/{concept_id}

```json
{
  "concept_id": "85189001",
  "active": true,
  "fsn": "Acute appendicitis (disorder)",
  "preferred_term": "Acute appendicitis",
  "preferred_term_language": "en",
  "preferred_terms": {"de": "Akute Appendizitis", "en": "Acute appendicitis"},
  "synonyms": {"fr": ["Appendicite aiguë"]},
  "semantic_tag": "disorder",
  "parents": ["74400008"],
  "ancestors": ["138875005", "404684003", "64572001", "74400008"],
  "snomed_version": "20250101"
}
```

`?language=de` selects the language of `preferred_term`. Unknown IDs return 404.

## GET /health

```json
{
  "status": "ok",
  "clif_version": "0.1.0",
  "snomed_version": "20250101",
  "snomed_edition": "SnomedCT_ClifDemoRF2_PRODUCTION_20250101T120000Z",
  "languages": ["en", "de", "nl", "fr"],
  "reranker": {"backend": "heuristic", "model": "bidirectional-coverage-v1"},
  "vector_retrieval": true,
  "calibrated": false
}
```

Before an index exists, `status` is `not_ready` with the reason. The API picks up a newly built index on the next request, so `docker compose up` can run before the import.

## Command line

Every endpoint has a CLI equivalent:

```bash
clif map "patient denies chest pain" --debug
clif map "appendectomy" --hierarchy procedure
clif candidates "Herzinfarkt" --language de
```

`clif --help` lists all commands, including `import-rf2`, `build-index`, `evaluate`, `export-laya-training` and `train-laya`.
