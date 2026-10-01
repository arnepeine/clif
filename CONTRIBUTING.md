# Contributing to Clif

Thanks for helping. Clif sits between clinical text and coded data, so a careful, well-tested change is worth more than a fast one.

## Two rules before anything else

1. **No patient data.** Not in issues, pull requests, test cases or screenshots. A mapping problem can always be shown with a short de-identified phrase ("appendicitis", "Z.n. Appendektomie").
2. **No SNOMED CT release content.** Don't commit releases, indexes, exported training data or fine-tuned weights. Tests use the hand-built demo release. Referencing an SCTID with its preferred term in a test is fine. See [docs/licensing.md](docs/licensing.md).

## Development setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/python -m pytest
```

The test suite builds the demo release in a temporary directory and runs in a few seconds. It needs no SNOMED licence, no network and no GPU. Laya and LLM judges are tested against mock servers.

To try your change against real SNOMED, import a release into its own data directory ([docs/snomed-data.md](docs/snomed-data.md)) and run the evaluation before and after:

```bash
CLIF_DATA_DIR=./data/intl clif evaluate data/eval/gps_eval_en.json --report before.json
```

## Good first contributions

| area | where | what helps |
|---|---|---|
| context cues | `clif/resources/context_cues.yaml` | negation, history and uncertainty cues for more languages (Spanish, Italian, Polish ...) |
| abbreviations | `clif/resources/abbreviations.yaml` | clinical abbreviations with priors and context cues; ambiguous ones are the valuable ones |
| evaluation cases | `data/eval/` | de-identified phrases with a verified expected SCTID or status, ideally from a clinical specialty we don't cover yet |
| ECL | `clif/terminology/hierarchy.py` | refinements and member-of, evaluated as predicates like the existing operators |
| judges | `clif/reranking/` | new backends implementing `ConceptReranker.rerank` |

## Pull requests

- One topic per pull request.
- Add or update tests. Bug fixes come with a test that fails before the fix.
- If you change retrieval, a judge or the decision logic, include the evaluation numbers before and after, from the demo set and, if you have one, a real edition. A higher accuracy that comes with a higher `wrong_match_rate` is a regression.
- Match the surrounding code: type hints, small functions, comments only where the code can't explain itself.
- Update the docs in `docs/` when behaviour or configuration changes, and add a line to [CHANGELOG.md](CHANGELOG.md).

## Reporting a wrong mapping

Use the "Wrong mapping" issue template. Please include:

- the phrase, de-identified
- the SNOMED edition and version (`snomed_version` in the response)
- the judge (`reranker.backend`)
- what Clif returned, and what you expected

The output of `clif candidates "<phrase>"` usually shows straight away whether retrieval or the judge is at fault.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
