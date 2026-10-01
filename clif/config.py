"""Runtime configuration. Every setting can be overridden with a CLIF_* env var."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

US_ENGLISH_REFSET = "900000000000509007"
GB_ENGLISH_REFSET = "900000000000508004"


class RetrievalWeights(BaseModel):
    """Weights of the hybrid retrieval score (env: CLIF_WEIGHTS__LEXICAL etc.)."""

    lexical: float = 0.40
    semantic: float = 0.35
    exact: float = 0.15
    prior: float = 0.10


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLIF_", env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("./data")
    db_filename: str = "terminology.sqlite"
    vector_dirname: str = "vectors"

    # --- languages ---------------------------------------------------------
    # Language code -> ordered list of language refset IDs. The first refset
    # decides the preferred term. Languages without an entry index every
    # active description with that language code.
    language_refsets: dict[str, list[str]] = Field(
        default_factory=lambda: {"en": [US_ENGLISH_REFSET, GB_ENGLISH_REFSET]}
    )
    # Languages whose descriptions are indexed. Empty list = all languages found.
    index_languages: list[str] = Field(default_factory=list)
    default_display_language: str = "en"
    # Descriptions of the input language are always searched; these are added
    # as a fallback (cross-lingual lexical + semantic retrieval).
    fallback_languages: list[str] = Field(default_factory=lambda: ["en"])

    # --- retrieval ---------------------------------------------------------
    candidate_count: int = 32
    description_pool: int = 200  # descriptions fetched per retrieval channel
    bm25_common_df_ratio: float = 0.02  # tokens in more descriptions than this are left out of OR queries
    fuzzy_min_similarity: float = 80.0  # rapidfuzz ratio for spelling correction against the vocabulary
    fuzzy_min_ratio: float = 60.0  # minimum lexical score for fuzzy-only hits
    retrieval_cache_size: int = 4096
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    # Soft semantic-category boost; hard filtering only above the certainty.
    category_weight: float = 0.15
    category_hard_filter_certainty: float = 0.97
    category_hard_filter_enabled: bool = False
    # Terminology prior per semantic tag (default 0.5 for unknown tags).
    semantic_tag_priors: dict[str, float] = Field(
        default_factory=lambda: {
            "disorder": 0.8,
            "finding": 0.8,
            "procedure": 0.75,
            "situation": 0.55,
            "body structure": 0.5,
            "substance": 0.55,
            "product": 0.5,
            "organism": 0.5,
            "observable entity": 0.55,
            "qualifier value": 0.3,
            "morphologic abnormality": 0.4,
            "navigational concept": 0.05,
            "foundation metadata concept": 0.0,
            "core metadata concept": 0.0,
            "namespace concept": 0.0,
            "linkage concept": 0.05,
            "attribute": 0.05,
        }
    )

    # --- embeddings / vector retrieval --------------------------------------
    embedder: Literal["none", "hashing", "sentence-transformers", "http"] = "hashing"
    embedder_model: str = ""  # local path / name for sentence-transformers, model for http
    embedder_url: str = ""  # OpenAI-compatible /v1/embeddings endpoint (on-prem)
    embedder_query_prefix: str = ""  # e.g. "query: " for e5 models
    embedder_passage_prefix: str = ""  # e.g. "passage: " for e5 models
    hashing_dim: int = 256
    vector_scope: Literal["all", "preferred"] = "all"
    embed_batch_size: int = 256

    # --- clinical context --------------------------------------------------
    # off: never use precoordinated context concepts (assertions always split off)
    # fixed_expressions: fixed phrases such as "no known drug allergies" are
    #   mapped as a whole (to situation concepts); everything else is split
    # prefer: also offer situation concepts when context cues were found
    situation_mode: Literal["off", "fixed_expressions", "prefer"] = "fixed_expressions"
    abbreviation_dominance: float = 0.9

    # --- reranking ---------------------------------------------------------
    # heuristic:  deterministic baseline (tests, offline)
    # laya:       Laya over HTTP (laya-serve, POST /v1/systemone, Jev-compatible)
    # laya-local: Laya in-process (pip install "clif[laya]"), e.g. a fine-tuned checkpoint
    # llm:        any OpenAI-compatible local LLM with JSON-schema constrained output
    reranker: Literal["heuristic", "laya", "laya-local", "llm"] = "heuristic"
    reranker_fallback: Literal["none", "heuristic"] = "none"
    reranker_timeout_s: float = 20.0
    reranker_concurrency: int = 8
    # Candidates handed to the reranker (retrieval still produces candidate_count).
    # Laya accuracy degrades beyond ~20 options (shared option token budget).
    rerank_candidates: int = 20

    laya_url: str = ""  # e.g. http://laya:8000 (the /v1/systemone path is appended)
    laya_api_key: str = ""  # LAYA_API_KEY of the server, sent as Bearer token
    # "" = pick by source language (english / multilingual); or english | multilingual | typed-decisions
    laya_model: str = ""
    # laya-local only: path or HF id of a (fine-tuned) checkpoint; "" = Laya Router with shipped checkpoints
    laya_checkpoint: str = ""
    laya_device: str = ""
    laya_head_max_len_min: int = 256
    laya_head_max_len_max: int = 1024
    laya_state_budget: int = 512
    laya_option_synonyms: int = 2  # synonyms rendered into each option description

    llm_url: str = ""  # OpenAI-compatible chat completions base URL, e.g. http://llm:8000/v1
    llm_model: str = ""
    llm_api_key: str = ""
    llm_use_json_schema: bool = True

    # --- decision ----------------------------------------------------------
    threshold_match: float = 0.60
    threshold_margin: float = 0.20
    threshold_no_match: float = 0.30
    threshold_plausible: float = 0.15
    calibration_file: str = ""  # JSON written by `clif evaluate --calibrate`

    # --- api ---------------------------------------------------------------
    max_text_length: int = 2000
    max_context_length: int = 20000
    max_batch_size: int = 500

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / self.vector_dirname


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
