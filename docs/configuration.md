# Configuration

Every setting is an environment variable with the prefix `CLIF_`, read from the environment or a `.env` file (template: [`.env.example`](../.env.example)). Nested values use `__`, for example `CLIF_WEIGHTS__LEXICAL=0.5`. Lists and dictionaries are JSON.

## Storage

| variable | default | |
|---|---|---|
| `CLIF_DATA_DIR` | `./data` | directory for the terminology database and vector index; use one directory per SNOMED edition |
| `CLIF_DB_FILENAME` | `terminology.sqlite` | |
| `CLIF_VECTOR_DIRNAME` | `vectors` | |

## Languages

| variable | default | |
|---|---|---|
| `CLIF_LANGUAGE_REFSETS` | `{"en": ["900000000000509007", "900000000000508004"]}` | language → ordered language refsets. The first refset decides the preferred term (here US English, then GB). Languages without an entry index all active descriptions of that language. Configured refsets that are missing from the release are ignored with a warning |
| `CLIF_INDEX_LANGUAGES` | `[]` | restrict indexing to these languages; empty means all |
| `CLIF_DEFAULT_DISPLAY_LANGUAGE` | `en` | language of returned preferred terms unless the request says otherwise |
| `CLIF_FALLBACK_LANGUAGES` | `["en"]` | always searched in addition to the input language |

For a German national edition, add its language refset: `CLIF_LANGUAGE_REFSETS={"en": [...], "de": ["<German refset id>"]}`.

## Retrieval

| variable | default | |
|---|---|---|
| `CLIF_CANDIDATE_COUNT` | `32` | distinct concepts kept after retrieval |
| `CLIF_DESCRIPTION_POOL` | `200` | descriptions fetched per channel and query variant before grouping |
| `CLIF_WEIGHTS__LEXICAL` | `0.40` | weight of lexical similarity |
| `CLIF_WEIGHTS__SEMANTIC` | `0.35` | weight of vector similarity (dropped when vectors are off) |
| `CLIF_WEIGHTS__EXACT` | `0.15` | bonus for exact and prefix matches |
| `CLIF_WEIGHTS__PRIOR` | `0.10` | terminology prior per semantic tag |
| `CLIF_SEMANTIC_TAG_PRIORS` | see `clif/config.py` | prior per semantic tag; metadata and navigational concepts are near zero |
| `CLIF_BM25_COMMON_DF_RATIO` | `0.02` | tokens in more than this share of descriptions are left out of BM25 OR queries |
| `CLIF_FUZZY_MIN_SIMILARITY` | `80` | rapidfuzz ratio needed to spell-correct a token against the vocabulary |
| `CLIF_FUZZY_MIN_RATIO` | `60` | minimum lexical score for hits found only through spelling correction |
| `CLIF_RETRIEVAL_CACHE_SIZE` | `4096` | LRU entries |
| `CLIF_CATEGORY_WEIGHT` | `0.15` | strength of the soft semantic-category factor |
| `CLIF_CATEGORY_HARD_FILTER_ENABLED` | `false` | allow removing candidates on a very confident category prediction |
| `CLIF_CATEGORY_HARD_FILTER_CERTAINTY` | `0.97` | |

## Vector search

| variable | default | |
|---|---|---|
| `CLIF_EMBEDDER` | `hashing` | `none`, `hashing` (character n-grams, no model needed), `sentence-transformers` (local model, `pip install "clif[embeddings]"`) or `http` (OpenAI-compatible `/v1/embeddings` service in your network) |
| `CLIF_EMBEDDER_MODEL` | | local model path or model name |
| `CLIF_EMBEDDER_URL` | | endpoint for `http` |
| `CLIF_EMBEDDER_QUERY_PREFIX` / `_PASSAGE_PREFIX` | | e.g. `"query: "` / `"passage: "` for e5 models |
| `CLIF_HASHING_DIM` | `256` | |
| `CLIF_VECTOR_SCOPE` | `all` | `all` synonyms or only `preferred` terms |
| `CLIF_EMBED_BATCH_SIZE` | `256` | |

Changing the embedder requires `build_index` again. Clif refuses to start with a vector index built by a different embedder.

## Clinical context

| variable | default | |
|---|---|---|
| `CLIF_SITUATION_MODE` | `fixed_expressions` | `off`: always split context off. `fixed_expressions`: phrases like "no known drug allergies" map to their situation concept. `prefer`: also offer situation concepts when context cues were found |
| `CLIF_ABBREVIATION_DOMINANCE` | `0.9` | posterior an abbreviation meaning needs to be resolved |

The cue lexicons are YAML files in `clif/resources/`. Edit them to add languages, cues or abbreviations; they're loaded at startup.

## Judge

| variable | default | |
|---|---|---|
| `CLIF_RERANKER` | `heuristic` | `heuristic`, `laya`, `laya-local` or `llm` |
| `CLIF_RERANKER_FALLBACK` | `none` | `heuristic`: answer with the rule-based judge if the primary fails, and flag it |
| `CLIF_RERANK_CANDIDATES` | `20` | candidates shown to the judge |
| `CLIF_RERANKER_TIMEOUT_S` | `20` | |
| `CLIF_RERANKER_CONCURRENCY` | `8` | parallel judge calls in batch mode |
| `CLIF_LAYA_URL` | | `laya-serve` base URL, e.g. `http://laya:8000` |
| `CLIF_LAYA_API_KEY` | | sent as bearer token |
| `CLIF_LAYA_MODEL` | | empty: chosen by language (`english` / `multilingual`) |
| `CLIF_LAYA_CHECKPOINT` | | `laya-local`: path to a fine-tuned checkpoint; empty uses Laya's router with the shipped checkpoints |
| `CLIF_LAYA_DEVICE` | | `mps`, `cuda`, `cpu`; empty picks the best available |
| `CLIF_LAYA_HEAD_MAX_LEN_MIN` / `_MAX` | `256` / `1024` | bounds for the per-request option token budget |
| `CLIF_LAYA_STATE_BUDGET` | `512` | tokens reserved for the clinical text and context |
| `CLIF_LAYA_OPTION_SYNONYMS` | `2` | extra terms shown per option |
| `CLIF_LLM_URL`, `CLIF_LLM_MODEL`, `CLIF_LLM_API_KEY` | | OpenAI-compatible chat endpoint for `llm` |
| `CLIF_LLM_USE_JSON_SCHEMA` | `true` | use `response_format: json_schema`; set false for servers that only support `json_object` |

## Decision

| variable | default | |
|---|---|---|
| `CLIF_THRESHOLD_MATCH` | `0.60` | minimum calibrated confidence for `MATCH` |
| `CLIF_THRESHOLD_MARGIN` | `0.20` | minimum lead over the second candidate |
| `CLIF_THRESHOLD_NO_MATCH` | `0.30` | below this, `NO_MATCH` |
| `CLIF_THRESHOLD_PLAUSIBLE` | `0.15` | a second candidate above this with a small margin makes the result `AMBIGUOUS` |
| `CLIF_CALIBRATION_FILE` | | JSON from `clif evaluate --calibrate`; overrides the thresholds above |

## API limits

| variable | default |
|---|---|
| `CLIF_MAX_TEXT_LENGTH` | `2000` |
| `CLIF_MAX_CONTEXT_LENGTH` | `20000` |
| `CLIF_MAX_BATCH_SIZE` | `500` |

## Docker

```bash
cp .env.example .env
docker compose up -d                    # API only, rule-based judge
docker compose --profile laya up -d     # API + laya-serve; set CLIF_RERANKER=laya in .env
docker compose run --rm clif python -m clif.import_rf2 /data/rf2/<release>.zip
docker compose run --rm clif python -m clif.build_index
```

- `./data` is mounted at `/data`.
- The API binds to `127.0.0.1` by default because it has no authentication of its own. Put it behind your API gateway before exposing it.
- The `laya` service builds from the pinned Laya release tag and is reachable only on the compose network. After the first start has cached the weights, set `HF_HUB_OFFLINE=1` so it never contacts the internet again.
- Build extras into the Clif image with `CLIF_EXTRAS=laya,faiss`.
