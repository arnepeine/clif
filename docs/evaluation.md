# Evaluation

A mapping system is only as trustworthy as its measurement. Clif's evaluation separates **retrieval failures** (the right concept never reached the judge) from **judge failures** (it was offered and not chosen). That tells you where to work.

## Running an evaluation

```bash
clif evaluate data/eval/gps_eval_en.json -v --report report.json
```

The command maps every case with the configured judge and SNOMED edition, then prints the metrics and, with `-v`, every failure with its category.

## Dataset format

JSON array or JSON Lines, one case per entry:

```json
[
  {"text": "acute appendicitis", "expected_concept_id": "85189001"},
  {"text": "patient denies chest pain", "expected_concept_id": "29857009", "expected_assertion": "NEGATED"},
  {"text": "AF", "expected_concept_id": "49436004",
   "context": "ECG demonstrates irregularly irregular rhythm."},
  {"text": "appendectomy", "expected_concept_id": "80146002", "allowed_hierarchies": ["procedure"]},
  {"text": "MS", "expected_status": "AMBIGUOUS"},
  {"text": "the weather is lovely today", "expected_status": "NO_MATCH"},
  {"text": "history of", "expected_status": "INSUFFICIENT_INFORMATION"}
]
```

| field | |
|---|---|
| `text` | required |
| `expected_concept_id` | the correct SCTID; implies `expected_status: MATCH` |
| `expected_status` | `MATCH`, `AMBIGUOUS`, `NO_MATCH` or `INSUFFICIENT_INFORMATION` (default `NO_MATCH` without a concept) |
| `context`, `language`, `allowed_hierarchies`, `ecl` | passed to `/map` |
| `expected_category` | broad category, e.g. `procedure`; otherwise derived from the expected concept |
| `expected_assertion`, `expected_temporality`, `expected_certainty` | checks the context attributes |

## Metrics

| metric | definition |
|---|---|
| `top1_accuracy` | `MATCH` with the expected concept, over cases expecting a match |
| `top5_recall` | expected concept among the judge's top five |
| `candidate_recall@32` | expected concept among retrieval's 32 candidates: **the retrieval ceiling** |
| `candidate_recall@20_offered_to_reranker` | expected concept among the candidates actually shown to the judge |
| `reranking_accuracy_given_present` | judge's first choice correct, over cases where the concept was offered: **the judge's own accuracy** |
| `mapping_accuracy_given_present` | as above, but after thresholds (status must be `MATCH`) |
| `no_match_accuracy`, `ambiguity_accuracy`, `insufficient_information_accuracy` | share of cases with that expected status that got it |
| `wrong_match_rate` | `MATCH` with a wrong concept, or `MATCH` where none was expected, over all cases: **the number that matters most** |
| `semantic_category_accuracy` | category hint vs. expected category |
| `context_attribute_accuracy` | assertion, temporality and certainty vs. expectations |
| `latency_ms` | p50, p95, p99 per mapping |

Every failure is labelled:

| label | meaning |
|---|---|
| `retrieval_miss` | not among the 32 candidates; fix synonyms, embeddings or the cue lexicon |
| `not_offered_to_reranker` | retrieved, but ranked below the judge's cut-off |
| `reranker_wrong_choice` | offered, judge preferred another option |
| `decision_ambiguous` / `decision_no_match` | judge was right, but below the confidence thresholds |
| `wrong_status` | expected `NO_MATCH` / `AMBIGUOUS` / `INSUFFICIENT_INFORMATION`, got something else |

## Calibration

```bash
clif evaluate my_validation_set.json --calibrate data/calibration.json
export CLIF_CALIBRATION_FILE=data/calibration.json
```

1. A logistic (Platt) calibrator is fitted over the judge's top probability, the retrieval score and the margin to the runner-up.
2. Thresholds are then grid-searched (match 0.30 to 0.95, margin 0 to 0.50) to maximize correct matches minus `--wrong-match-penalty` (default 4) times wrong matches.
3. Ties go to the stricter setting.

Calibrate on a validation set and report results on a separate test set. Calibrating and reporting on the same set overstates performance.

## Results

### Test set

[`data/eval/gps_eval_en.json`](../data/eval/gps_eval_en.json): 64 English cases.

- **55 expect a concept.** They cover specificity (`appendicitis`), synonyms (`heart attack`), British spelling (`anaemia`), typos (`atrial fibrilation`), negation, history, uncertainty, family history, abbreviations with context, a hierarchy restriction, drugs, organisms and observables.
- **9 expect a refusal.** Three ambiguous abbreviations, four non-clinical phrases, two phrases without content.
- **Every expected SCTID was verified** active in the SNOMED release it was tested against.
- **All 64 phrases were excluded** from Laya training data.

### SNOMED CT International Edition 20261001

| | rule-based | Laya untrained | Laya fine-tuned (v2) |
|---|---|---|---|
| top-1 accuracy | 85.5% | 56.4% | **90.9%** |
| top-5 recall | 100% | 96.4% | 100% |
| candidate recall @ 32 and @ 20 | 100% | 100% | 100% |
| judge accuracy given present | 98.2% | 81.8% | 96.4% |
| no-match accuracy | 50% | 75% | **100%** |
| ambiguity accuracy | 100% | 100% | 100% |
| wrong-match rate | 0% | 1.6% | **0%** |
| context attributes | 100% | 100% | 100% |
| latency p50 / p95 | 15 / 35 ms | 92 / 143 ms | 53 / 85 ms |

- **Rule-based judge:** almost always ranks the right concept first (98%), but its distributions are flat when SNOMED offers many near-duplicates, so it often falls below the match threshold and answers `AMBIGUOUS`.
- **Fine-tuned Laya:** confident when it's right and says "none" when it's unsure.
- **Neither** returned a wrong code.

### Global Patient Set 20260101 (no synonyms)

| | rule-based | Laya untrained |
|---|---|---|
| top-1 accuracy | 78.2% | 76.4% |
| candidate recall @ 32 | 89.1% | 89.1% |
| wrong-match rate | 6.3% | 7.8% |

Without synonyms, phrases like "heart attack" or "high blood pressure" never reach the judge. Both judges then pick something lexically close but wrong ("Fear of having a heart attack", "Blood pressure"). This is a data limitation of the GPS, and the reason to use the International Edition or a semantic embedding model.

Raw reports: [`docs/results/`](results/).

## What these numbers don't tell you

- **64 cases is small.** One case is 1.6 percentage points.
- **The phrases are close to SNOMED wording.** Real documentation has run-on sentences, local abbreviations, misspellings and mixed languages. Expect lower numbers there.
- **English only.** Non-English performance depends on the national edition you import.
- **The test set was written by the same team that built the system.** An independent set is the real test.

## Building your own test set

1. Collect a few hundred phrases from the documentation you want to code: problem lists, discharge diagnoses, procedure notes. Include the messy ones.
2. **Remove every patient identifier** before the phrases leave the clinical system. A phrase like "appendicitis" needs no patient context.
3. Have a clinician or terminologist assign the expected SCTID, or `NO_MATCH` / `AMBIGUOUS` where that is the honest answer.
4. Split it: half for calibration, half for reporting.
5. Exclude it from training (`clif export-laya-training --exclude-eval your_set.json`).
6. Track `wrong_match_rate` over time. Accuracy can be traded for coverage; wrong codes can't.
