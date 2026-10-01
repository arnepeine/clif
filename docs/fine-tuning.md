# Fine-tuning Laya on SNOMED CT

Laya's base checkpoints are general-purpose decision models. Its own documentation reports near-chance zero-shot accuracy on new decision types and recommends fine-tuning. That held for SNOMED too: untrained, Laya picked the right concept in 56% of our test phrases, below the rule-based judge (85%). Fine-tuned, it reached 91%.

This document explains how the training data is generated, how training works, what went wrong on the way, and how to reproduce it.

## The idea: SNOMED supervises itself

Every SNOMED synonym is a phrase whose correct concept is known. "Heart attack" is a description of *Myocardial infarction*. For each sampled synonym, Clif:

1. runs its **real retrieval** for the phrase, exactly as in production,
2. renders the **exact question** the Laya judge receives in production: the phrase plus about 20 candidate concepts, plus "none of these" and "can't decide",
3. records the correct answer.

Because training questions and production questions are built by the same code, the model learns the task it will actually face, including the near-duplicates real retrieval returns ("Fear of having a heart attack" next to "Myocardial infarction").

### Example types

| type | how it's made | share in our run |
|---|---|---|
| positive | correct concept among the options | 25,285 |
| no match | correct concept removed from the options; answer: none | 4,441 |
| retrieval miss | retrieval didn't find the concept; answer: none | 274 |
| non-clinical negatives | everyday sentences, recombinations and pseudo-words; answer: none | 1,500 |
| specificity (subset of positives) | a parent concept's term with its children among the options | 7,858 |

### Sampling

SNOMED is about a quarter organisms and substances. Uniform sampling would teach the model bacteria names more than diagnoses, so sampling is weighted by semantic tag:

| weight | semantic tags |
|---|---|
| 1.0 | disorder, finding, procedure |
| 0.8 | situation |
| 0.6 | observable entity, regime/therapy |
| 0.5 | body structure |
| 0.4 | morphologic abnormality |
| 0.35 | substance |
| 0.3 | product, specimen, physical object, event |
| 0.15 | qualifier value |
| 0.1 | organism and everything else |

### Test-set exclusion

`--exclude-eval <dataset>` removes every phrase of an evaluation set from the training queries, for SNOMED examples and negatives alike. Without it, a test phrase that's also a SNOMED synonym ("heart attack") would be answered from memory.

## Training

`clif train-laya` is a single-device port of the training loop in Laya's fine-tuning notebook (v0.3.21):

- **Objective:** policy gradient against Laya's proper scoring reward (group-relative baseline, 4 sampled logit perturbations, σ annealed 0.4 → 0.1) plus soft cross-entropy.
- **Learning rates:** 2.5e-5 for the encoder, 1e-4 for the decision head, AdamW, cosine schedule.
- **Batching:** micro-batch 8, gradient accumulation 4 (effective batch 32), bf16 autocast on MPS and CUDA.
- **Token budget:** each example is tokenized with the `head_max_len` and `max_len` stored in its row, the same budget Clif sends at inference. The notebook uses one fixed budget.
- **Length grouping:** examples of similar length share a batch, which cuts padding.
- **Held-out slice:** 5% (at most 3,000 examples) is reported after every epoch and used for the final temperature fit.
- **Rolling checkpoints:** model, optimizer, scheduler and position in the data are saved every `--save-every` optimizer steps. `--resume` continues mid-epoch.
- **Output:** a regular Laya checkpoint (`laya.load(path)` works), with training metadata in `rl_agent_config.json`.

Base model: `convaiinnovations/laya`, subfolder `multilingual` (mmBERT-base, 322M parameters), so the same checkpoint can later learn German.

## Results

All runs on an Apple M5 Pro (MPS), SNOMED CT International Edition 20261001. "Clif test" is the 64-phrase evaluation set with verified codes, excluded from training.

| run | data | compute | held-out SNOMED | Clif test: correct | rejects nonsense | confidently wrong |
|---|---|---|---|---|---|---|
| base, untrained | | | 38% | 56% | 75% | 1.6% |
| pilot | 1,425 examples, 1 epoch, no negatives | 3 min | 87% | 87% | 25% | 3.1% |
| v1 | 31,500 examples, 2 epochs | ~1.7 h | 98% | **2%** | 100% | 0% |
| v2 | v1 + 1 epoch with casing balanced | 48 min | 97.5% | **91%** | 100% | 0% |
| rule-based judge, for comparison | | | | 85% | 50% | 0% |

The raw reports are in [`docs/results/`](results/).

## What went wrong, and what it teaches

### The pilot: never seeing unrelated text

Trained only on real SNOMED phrases, the pilot never saw an example where "none of these" was right for unrelated text. It matched "parking garage level three" to a concept and rejected easy cases like "pneumonia".

Fix: non-clinical negative examples.

### v1: the casing shortcut

v1 scored 98% on held-out SNOMED questions and then answered "none of these" for almost every test phrase. The cause:

- SNOMED terms are capitalized ("Appendicitis").
- The negative examples were everyday sentences, all lowercase.
- So "lowercase" perfectly predicted "none", and the model learned that instead of meaning.
- The test phrases were typed in lowercase, like real input.

| input to v1 | answer |
|---|---|
| `Appendicitis` | *Appendicitis*, 99% |
| `appendicitis` | none of these, 99% |

The held-out score didn't reveal it, because the held-out slice had the same bias as the training data. On a held-out slice with varied casing, v1 scored 62%.

Fix: `vary_case` gives every example type the same casing distribution (45% lowercase, 10% uppercase, 45% capitalized; short acronyms kept). v2 continued from v1 for one epoch on the corrected data.

**Lesson:** a held-out split from the same generator can't catch shortcuts the generator itself introduces. Always evaluate on independently written phrases, and compare the class distributions of every surface feature you can think of (casing, length, punctuation, language) between positives and negatives.

### v2's remaining misses

All five were cautious, not wrong:

- *aortic stenosis* and *influenza* came back as none, although the right concept was offered;
- the long sentence "patient has severe pain in the right lower abdomen";
- "AF" and "MS" with context. The SNOMED-generated data contains almost no abbreviations, so abbreviation handling still rests on Clif's resolver, not on the model.

## Reproducing

```bash
export CLIF_DATA_DIR=./data/intl
python -m clif.import_rf2 SnomedCT_InternationalRF2_PRODUCTION_20261001T120000Z.zip
python -m clif.build_index

clif export-laya-training --out data/intl/laya_train.jsonl --n 30000 --negatives 1500 \
  --exclude-eval data/eval/gps_eval_en.json --exclude-eval data/eval/demo_eval.json
clif train-laya --data data/intl/laya_train.jsonl --out models/laya-snomed --epochs 2

CLIF_RERANKER=laya-local CLIF_LAYA_CHECKPOINT=models/laya-snomed \
  clif evaluate data/eval/gps_eval_en.json -v
```

The export now includes casing variation, so a fresh two-epoch run corresponds to v2 without the intermediate v1 step.

### Training on every synonym

`scripts/train_full_laya.sh` runs the whole corpus (659,026 English synonyms in the International Edition):

1. a parallel export (`--all --workers 10`, about one hour)
2. 10,000 negatives
3. one epoch with rolling checkpoints (about 18 hours on an M5 Pro)
4. evaluation

If interrupted, run it again with `RESUME=1`. On a laptop, keep it plugged in with the lid open; the script holds a keep-awake assertion while it runs.

### Hardware

| | |
|---|---|
| tested | Apple M5 Pro, 64 GB, PyTorch MPS |
| speed | about 10 training examples per second |
| other hardware | CUDA works with the same script and is considerably faster |

## Using the checkpoint

```bash
CLIF_RERANKER=laya-local CLIF_LAYA_CHECKPOINT=models/laya-snomed uvicorn clif.api:app
```

Recalibrate the decision thresholds after every model change (`clif evaluate <validation set> --calibrate data/calibration.json`).

Model weights trained on SNOMED content are derived from licensed material. Check [licensing](licensing.md) before sharing them.
