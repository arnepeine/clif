# Licensing

## Clif

Clif's code is licensed under the [Apache License 2.0](../LICENSE). You may use, modify and redistribute it, including commercially, under the terms of that licence.

## SNOMED CT

SNOMED CT is owned by SNOMED International and licensed separately. Clif contains no SNOMED CT release. To use Clif with real SNOMED content you need a licence for the release you import:

- **SNOMED member countries** (Germany, the UK, the US, the Netherlands and many more): an affiliate licence is free. Apply through [MLDS](https://mlds.ihtsdotools.org) or your national release centre.
- **The [Global Patient Set](https://www.snomed.org/gps)** is free for everyone under CC BY-ND 4.0, after registration.

### What this repository contains

| content | where | basis |
|---|---|---|
| about 100 SCTIDs with FSN and preferred term | `clif/devtools/demo_release.py`, `tests/`, `data/eval/` | identifiers and terms as published in the Global Patient Set (CC BY-ND 4.0), attributed in [NOTICE](../NOTICE) |
| synonyms and German, Dutch and French translations in the demo release | `clif/devtools/demo_release.py` | written by hand for testing; not taken from any SNOMED extension or national edition |
| synthetic concept IDs | `clif/devtools/demo_release.py` (`fake(n)`) | invented, check-digit valid, in a range no real release uses |
| evaluation phrases with expected SCTIDs | `data/eval/` | hand-written phrases; SCTIDs referenced, not SNOMED descriptions |
| evaluation reports | `docs/results/` | metrics and failing cases with SCTIDs |

### What you must not publish

| artefact | why |
|---|---|
| imported releases and built indexes (`data/*`) | they contain the licensed terminology |
| exported training data (`*.jsonl` from `export-laya-training`) | every example contains SNOMED descriptions |
| fine-tuned model weights (`models/`) | trained on licensed SNOMED content; whether they count as a derivative work under your licence is a question for SNOMED International or your national release centre |

`.gitignore` excludes `data/` and `models/` for exactly this reason. If you want to share a fine-tuned checkpoint, ask SNOMED International or your national release centre first. What you can always share is the recipe: the code, the commands and the evaluation reports, so others can train the same model under their own licence.

This page is a practical summary, not legal advice.

## Dependencies

| dependency | licence | use |
|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | Apache 2.0 | optional judge (`clif[laya]`), not vendored |
| FastAPI, Starlette, Pydantic, Uvicorn | MIT / BSD | API |
| NumPy | BSD | vectors |
| RapidFuzz | MIT | fuzzy matching |
| httpx | BSD | HTTP clients |
| PyYAML | MIT | lexicons |
| sentence-transformers | Apache 2.0 | optional embeddings |
| faiss-cpu | MIT | optional vector index |

Embedding or Laya checkpoints you download come with their own licences; check them before redistribution.
