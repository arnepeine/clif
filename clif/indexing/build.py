"""Builds the search index from an imported RF2 database.

Creates, inside the same SQLite file:
- ``concept_meta``: active concepts with FSN + semantic tag
- ``concept_pt``: preferred term per concept and language
- ``is_a``: active inferred IS-A edges between active concepts
- ``search_terms``: every searchable active description with normalized forms
- ``fts_terms``: FTS5 BM25 index (unicode61, diacritics removed)
- ``fts_terms_vocab``: token vocabulary with document frequencies (fuzzy
  spelling correction and BM25 query planning)

and a vector index (``data/vectors``) if an embedder is configured.

Usage:
    python -m clif.build_index
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from clif.config import Settings
from clif.retrieval.vector import make_embedder, write_vector_index
from clif.terminology.rf2 import FSN_TYPE, IS_A, PREFERRED, SYNONYM_TYPE
from clif.text import fold, normalize, strip_semantic_tag

log = logging.getLogger(__name__)

# Bump when the derived tables change; the runtime refuses indexes from other versions.
INDEX_SCHEMA_VERSION = "2"

DERIVED_SCHEMA = """
DROP TABLE IF EXISTS fts_trigram_vocab;
DROP TABLE IF EXISTS fts_trigram;
DROP TABLE IF EXISTS fts_terms_vocab;
DROP TABLE IF EXISTS fts_terms;
DROP TABLE IF EXISTS search_terms;
DROP TABLE IF EXISTS concept_meta;
DROP TABLE IF EXISTS concept_pt;
DROP TABLE IF EXISTS is_a;
CREATE TABLE concept_meta(
    concept_id TEXT PRIMARY KEY, fsn TEXT, semantic_tag TEXT, effective_time TEXT) WITHOUT ROWID;
CREATE TABLE concept_pt(
    concept_id TEXT, language TEXT, description_id TEXT, term TEXT,
    PRIMARY KEY(concept_id, language)) WITHOUT ROWID;
CREATE TABLE is_a(child TEXT NOT NULL, parent TEXT NOT NULL, PRIMARY KEY(child, parent)) WITHOUT ROWID;
CREATE INDEX ix_isa_parent ON is_a(parent);
CREATE TABLE search_terms(
    rowid INTEGER PRIMARY KEY,
    description_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    language TEXT NOT NULL,
    type TEXT NOT NULL,           -- FSN | SYN
    term TEXT NOT NULL,           -- original term (FSN without semantic tag)
    norm TEXT NOT NULL,
    folded TEXT NOT NULL,
    is_pt INTEGER NOT NULL,       -- preferred term in its language
    acceptability TEXT            -- refset:P|A list
);
"""

DERIVED_INDEXES = """
CREATE INDEX ix_st_norm ON search_terms(norm);
CREATE INDEX ix_st_folded ON search_terms(folded);
CREATE INDEX ix_st_concept ON search_terms(concept_id);
CREATE VIRTUAL TABLE fts_terms USING fts5(
    folded, content='search_terms', content_rowid='rowid', tokenize="unicode61 remove_diacritics 2");
INSERT INTO fts_terms(fts_terms) VALUES('rebuild');
CREATE VIRTUAL TABLE fts_terms_vocab USING fts5vocab(fts_terms, 'row');
"""


@dataclass
class BuildStats:
    concepts: int
    search_terms: int
    vectors: int
    seconds: float


def _eligible(lang: str, acc: dict[str, str], settings: Settings, refsets_by_lang: dict[str, list[str]]) -> bool:
    if settings.index_languages and lang not in settings.index_languages:
        return False
    refsets = refsets_by_lang.get(lang)
    if refsets:
        return any(r in acc for r in refsets)
    return True


def _pt_rank(lang: str, acc: dict[str, str], refsets_by_lang: dict[str, list[str]]) -> int | None:
    """Lower is better. None = not a preferred term candidate."""
    refsets = refsets_by_lang.get(lang)
    if refsets:
        for i, r in enumerate(refsets):
            if acc.get(r) == PREFERRED:
                return i
        return None
    return 0 if PREFERRED in acc.values() else None


def build_index(settings: Settings) -> BuildStats:
    t0 = time.time()
    conn = sqlite3.connect(str(settings.db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    if not conn.execute("SELECT name FROM sqlite_master WHERE name='concepts'").fetchone():
        raise RuntimeError(f"No imported RF2 data in {settings.db_path}. Run `python -m clif.import_rf2` first.")
    conn.executescript(DERIVED_SCHEMA)

    log.info("Building is_a hierarchy")
    conn.execute(
        """INSERT OR IGNORE INTO is_a(child, parent)
           SELECT r.source_id, r.destination_id FROM relationships r
           JOIN concepts s ON s.id = r.source_id AND s.active = 1
           JOIN concepts d ON d.id = r.destination_id AND d.active = 1
           WHERE r.active = 1 AND r.type_id = ?""",
        (IS_A,),
    )

    log.info("Selecting searchable descriptions")
    cur = conn.execute(
        """SELECT d.id, d.concept_id, d.language_code, d.type_id, d.term, c.effective_time,
                  (SELECT group_concat(l.refset_id || ':' || l.acceptability_id)
                     FROM langrefset l WHERE l.referenced_component_id = d.id AND l.active = 1)
           FROM descriptions d JOIN concepts c ON c.id = d.concept_id
           WHERE d.active = 1 AND c.active = 1 AND d.type_id IN (?, ?)
           ORDER BY d.concept_id""",
        (FSN_TYPE, SYNONYM_TYPE),
    )

    # Only configured refsets that actually exist in the release count. Slim packages
    # (e.g. the Global Patient Set) may lack them; then every active description of
    # that language is indexed instead of silently indexing none.
    present = {r[0] for r in conn.execute("SELECT DISTINCT refset_id FROM langrefset WHERE active = 1")}
    refsets_by_lang: dict[str, list[str]] = {}
    for lang, refsets in settings.language_refsets.items():
        usable = [r for r in refsets if r in present]
        if refsets and not usable:
            log.warning("none of the configured %s language refsets %s are in the release; indexing all "
                        "active '%s' descriptions", lang, refsets, lang)
        refsets_by_lang[lang] = usable
    en_refsets = refsets_by_lang.get("en", [])
    first_syn: dict[tuple[str, str], tuple[str, str]] = {}  # PT fallback when no refset marks one
    fsn_best: dict[str, tuple[int, str, str]] = {}  # concept -> (rank, fsn, effective_time)
    pt_best: dict[tuple[str, str], tuple[int, str, str]] = {}  # (concept, lang) -> (rank, desc id, term)
    rows: list[tuple] = []
    for did, cid, lang, type_id, term, cet, acc_s in cur:
        acc = dict(p.split(":", 1) for p in acc_s.split(",")) if acc_s else {}
        if type_id == FSN_TYPE:
            # FSN: prefer English, preferred in the first configured en refset.
            rank = 100
            if lang == "en":
                rank = 50
                for i, r in enumerate(en_refsets):
                    if acc.get(r) == PREFERRED:
                        rank = i
                        break
            if cid not in fsn_best or rank < fsn_best[cid][0]:
                fsn_best[cid] = (rank, term, cet)
        if not _eligible(lang, acc, settings, refsets_by_lang):
            continue
        is_syn = type_id == SYNONYM_TYPE
        text = strip_semantic_tag(term)[0] if not is_syn else term
        acc_str = ",".join(f"{r}:{'P' if a == PREFERRED else 'A'}" for r, a in sorted(acc.items()))
        rows.append([did, cid, lang, "SYN" if is_syn else "FSN", text, acc_str])
        if is_syn:
            first_syn.setdefault((cid, lang), (did, term))
            rank = _pt_rank(lang, acc, refsets_by_lang)
            if rank is not None and ((cid, lang) not in pt_best or rank < pt_best[(cid, lang)][0]):
                pt_best[(cid, lang)] = (rank, did, term)

    for key, (did, term) in first_syn.items():
        if key not in pt_best:
            pt_best[key] = (99, did, term)
    log.info("Writing %d search terms", len(rows))
    pt_ids = {v[1] for v in pt_best.values()}
    conn.executemany(
        "INSERT INTO search_terms(description_id, concept_id, language, type, term, norm, folded, is_pt, "
        "acceptability) VALUES(?,?,?,?,?,?,?,?,?)",
        (
            (did, cid, lang, typ, text, normalize(text), fold(normalize(text)), int(did in pt_ids), acc)
            for did, cid, lang, typ, text, acc in rows
        ),
    )
    conn.executemany(
        "INSERT INTO concept_meta VALUES(?,?,?,?)",
        ((cid, fsn, strip_semantic_tag(fsn)[1], et) for cid, (_, fsn, et) in fsn_best.items()),
    )
    conn.executemany(
        "INSERT INTO concept_pt VALUES(?,?,?,?)",
        ((cid, lang, did, term) for (cid, lang), (_, did, term) in pt_best.items()),
    )
    conn.commit()
    log.info("Building FTS5 indexes")
    conn.executescript(DERIVED_INDEXES)
    conn.execute("INSERT OR REPLACE INTO meta VALUES('index_built_at', datetime('now'))")
    conn.execute("INSERT OR REPLACE INTO meta VALUES('index_schema', ?)", (INDEX_SCHEMA_VERSION,))
    conn.commit()

    n_vec = 0
    embedder = make_embedder(settings)
    if embedder is not None:
        where = "WHERE type = 'SYN'" if settings.vector_scope == "all" else "WHERE is_pt = 1"
        vrows = conn.execute(f"SELECT rowid, term FROM search_terms {where} ORDER BY rowid").fetchall()
        log.info("Embedding %d terms with %s", len(vrows), embedder.identity)
        ids = np.fromiter((r[0] for r in vrows), dtype=np.int64, count=len(vrows))
        mat = np.zeros((len(vrows), embedder.dim), dtype=np.float32)
        bs = settings.embed_batch_size
        for start in range(0, len(vrows), bs):
            mat[start : start + bs] = embedder.embed([r[1] for r in vrows[start : start + bs]], kind="passage")
            if start and start % (bs * 200) == 0:
                log.info("  %d / %d", start, len(vrows))
        write_vector_index(settings.vector_dir, embedder.identity, embedder.dim, ids, mat)
        conn.execute("INSERT OR REPLACE INTO meta VALUES('embedder', ?)", (embedder.identity,))
        n_vec = len(vrows)
    conn.commit()
    conn.execute("ANALYZE")
    conn.close()
    return BuildStats(concepts=len(fsn_best), search_terms=len(rows), vectors=n_vec, seconds=time.time() - t0)
