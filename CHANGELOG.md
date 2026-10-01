# Changelog

All notable changes are recorded here. The project follows [semantic versioning](https://semver.org).

## 0.1.0 · 2026-10

First public release.

### Pipeline
- Two-stage mapping: SNOMED-wide hybrid retrieval to 32 candidates, constrained-choice judge over 20, terminology validation, calibrated decision.
- Statuses `MATCH`, `AMBIGUOUS`, `NO_MATCH`, `INSUFFICIENT_INFORMATION` with machine-readable reasons.
- Context analysis in English, German, Dutch and French: negation, temporality, certainty, experiencer, fillers, fixed expressions.
- Context-sensitive abbreviation resolution with priors and multiplicative cue evidence.
- Multilingual retrieval with separate source, matched-description and display languages.
- Hierarchy restrictions by name or SCTID, and an ECL subset (`<<`, `<`, `<!`, `>>`, `>`, `>!`, `AND`, `OR`, `MINUS`).

### Retrieval
- Exact, accent-folded, prefix, planned BM25 (SQLite FTS5), vocabulary spelling correction, vector search (hashing, sentence-transformers or HTTP embeddings, optional FAISS).
- Retrieval p50 of 24 ms on a 1.4M-description benchmark.

### Judges
- Rule-based judge (default and fallback).
- Laya over HTTP (`laya-serve`) and in-process, with per-request token budgets and option-collapse detection.
- OpenAI-compatible local LLMs with JSON-schema constrained output.
- Answers that weren't offered are always rejected.

### Terminology
- RF2 importer for Snapshot, Full and Delta, zip or directory, layered editions.
- Global Patient Set importer.
- Index schema versioning.

### Training and evaluation
- SNOMED-derived Laya training data with semantic-tag weighting, negatives, casing variation and evaluation-set exclusion.
- Single-device Laya training with rolling checkpoints and resume.
- Evaluation separating retrieval recall from judge accuracy; Platt calibration and threshold search.
- Fine-tuned Laya: 91% top-1, 0% wrong matches on the 64-phrase English test set (International Edition 20261001).

### API and deployment
- `POST /map`, `/map/batch`, `/candidates`; `GET /concepts/{id}`, `/health`.
- Docker image and compose setup with an optional `laya-serve` service.
