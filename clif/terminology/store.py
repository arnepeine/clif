"""Read-only runtime access to the terminology database.

One SQLite connection per thread (read-only). Concept metadata and hierarchy
traversals are cached in memory.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Description:
    rowid: int
    description_id: str
    concept_id: str
    language: str
    type: str
    term: str
    norm: str
    folded: str
    is_pt: bool


@dataclass
class ConceptRecord:
    concept_id: str
    fsn: str
    semantic_tag: str | None
    active: bool
    preferred_terms: dict[str, str]  # language -> PT
    synonyms: dict[str, list[str]]  # language -> active synonyms (excluding PT)
    parents: list[str]
    effective_time: str | None = None
    ancestors: list[str] = field(default_factory=list)

    def preferred_term(self, language: str, fallback: str = "en") -> tuple[str, str]:
        """Return (term, language actually used)."""
        for lang in (language, fallback, "en"):
            if lang in self.preferred_terms:
                return self.preferred_terms[lang], lang
        if self.preferred_terms:
            lang, term = next(iter(self.preferred_terms.items()))
            return term, lang
        from clif.text import strip_semantic_tag

        return strip_semantic_tag(self.fsn)[0], "en"


_DESC_COLS = "rowid, description_id, concept_id, language, type, term, norm, folded, is_pt"


def _desc(row) -> Description:
    return Description(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], bool(row[8]))


class TerminologyStore:
    def __init__(self, db_path: Path, cache_size: int = 65536):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"terminology database not found: {db_path}")
        self.db_path = Path(db_path)
        self._local = threading.local()
        conn = self._conn()
        if not conn.execute("SELECT name FROM sqlite_master WHERE name='search_terms'").fetchone():
            raise RuntimeError(f"search index missing in {db_path}; run `python -m clif.build_index`")
        self.meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        from clif.indexing.build import INDEX_SCHEMA_VERSION

        if self.meta.get("index_schema") != INDEX_SCHEMA_VERSION:
            raise RuntimeError(
                f"search index in {db_path} has schema {self.meta.get('index_schema', '1')}, this Clif needs "
                f"{INDEX_SCHEMA_VERSION}; rerun `python -m clif.build_index` (no re-import needed)"
            )
        self.languages = [r[0] for r in conn.execute("SELECT DISTINCT language FROM search_terms")]
        self.concept = lru_cache(maxsize=cache_size)(self._concept)  # type: ignore[method-assign]
        self.ancestors = lru_cache(maxsize=cache_size)(self._ancestors)  # type: ignore[method-assign]
        self.total_terms = conn.execute("SELECT count(*) FROM search_terms").fetchone()[0]
        # False for releases without relationships (Global Patient Set): hierarchy
        # restrictions then fall back to semantic tags, ECL is refused.
        self.has_hierarchy = conn.execute("SELECT 1 FROM is_a LIMIT 1").fetchone() is not None
        self._vocab_df: dict[str, int] | None = None
        self._vocab_by_len: dict[int, list[str]] | None = None
        self._vocab_lock = threading.Lock()

    # -- connection ----------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            c.execute("PRAGMA query_only=1")
            c.execute("PRAGMA cache_size=-65536")
            c.execute("PRAGMA mmap_size=1073741824")
            self._local.conn = c
        return c

    @property
    def snomed_version(self) -> str:
        return self.meta.get("snomed_version", "unknown")

    @property
    def snomed_edition(self) -> str:
        return self.meta.get("snomed_edition", "")

    # -- concepts --------------------------------------------------------------
    def _concept(self, concept_id: str) -> ConceptRecord | None:
        c = self._conn()
        row = c.execute(
            "SELECT m.fsn, m.semantic_tag, m.effective_time FROM concept_meta m WHERE m.concept_id = ?",
            (concept_id,),
        ).fetchone()
        if row is None:
            active = c.execute("SELECT active FROM concepts WHERE id = ?", (concept_id,)).fetchone()
            if active is None:
                return None
            return ConceptRecord(concept_id, "", None, bool(active[0]), {}, {}, [])
        pts = dict(c.execute("SELECT language, term FROM concept_pt WHERE concept_id = ?", (concept_id,)))
        syns: dict[str, list[str]] = {}
        for lang, term in c.execute(
            "SELECT language, term FROM search_terms WHERE concept_id = ? AND type = 'SYN' AND is_pt = 0 "
            "ORDER BY length(term)",
            (concept_id,),
        ):
            if term not in syns.setdefault(lang, []):
                syns[lang].append(term)
        parents = [r[0] for r in c.execute("SELECT parent FROM is_a WHERE child = ? ORDER BY parent", (concept_id,))]
        return ConceptRecord(concept_id, row[0], row[1], True, pts, syns, parents, row[2])

    def is_active(self, concept_id: str) -> bool:
        rec = self.concept(concept_id)
        return bool(rec and rec.active and rec.fsn)

    def full_record(self, concept_id: str) -> ConceptRecord | None:
        rec = self.concept(concept_id)
        if rec is None:
            return None
        return replace(rec, ancestors=sorted(self.ancestors(concept_id)))

    # -- hierarchy -----------------------------------------------------------
    def _ancestors(self, concept_id: str) -> frozenset[str]:
        rows = self._conn().execute(
            """WITH RECURSIVE anc(id) AS (
                   SELECT parent FROM is_a WHERE child = ?
                   UNION SELECT i.parent FROM is_a i JOIN anc ON i.child = anc.id)
               SELECT id FROM anc""",
            (concept_id,),
        )
        return frozenset(r[0] for r in rows)

    def is_descendant_or_self(self, concept_id: str, ancestor_id: str) -> bool:
        return concept_id == ancestor_id or ancestor_id in self.ancestors(concept_id)

    def children(self, concept_id: str) -> list[str]:
        return [r[0] for r in self._conn().execute("SELECT child FROM is_a WHERE parent = ?", (concept_id,))]

    def refset_members(self, refset_id: str) -> frozenset[str]:
        rows = self._conn().execute(
            "SELECT referenced_component_id FROM langrefset WHERE refset_id = ? AND active = 1", (refset_id,)
        )
        return frozenset(r[0] for r in rows)

    # -- description lookups ---------------------------------------------------
    def _lang_clause(self, languages: list[str] | None) -> tuple[str, list]:
        if not languages:
            return "", []
        return f" AND s.language IN ({','.join('?' * len(languages))})", list(languages)

    def exact(self, norm: str, folded: str, languages: list[str] | None, limit: int = 200) -> list[Description]:
        lc, lp = self._lang_clause(languages)
        rows = self._conn().execute(
            f"SELECT {_DESC_COLS} FROM search_terms s WHERE (s.norm = ? OR s.folded = ?){lc} LIMIT ?",
            [norm, folded, *lp, limit],
        )
        return [_desc(r) for r in rows]

    def prefix(self, folded: str, languages: list[str] | None, limit: int = 50) -> list[Description]:
        lc, lp = self._lang_clause(languages)
        rows = self._conn().execute(
            f"SELECT {_DESC_COLS} FROM search_terms s WHERE s.folded >= ? AND s.folded < ?{lc} "
            f"ORDER BY length(s.folded) LIMIT ?",
            [folded, folded + "\U0010ffff", *lp, limit],
        )
        return [_desc(r) for r in rows]

    def bm25(self, match: str, languages: list[str] | None, limit: int) -> list[tuple[Description, float]]:
        lc, lp = self._lang_clause(languages)
        rows = self._conn().execute(
            f"SELECT {', '.join('s.' + c.strip() for c in _DESC_COLS.split(','))}, bm25(fts_terms) "
            f"FROM fts_terms JOIN search_terms s ON s.rowid = fts_terms.rowid "
            f"WHERE fts_terms MATCH ?{lc} ORDER BY bm25(fts_terms) LIMIT ?",
            [match, *lp, limit],
        )
        return [(_desc(r), -float(r[9])) for r in rows]

    # -- token vocabulary (for spelling correction and BM25 planning) -----------
    def _load_vocab(self) -> None:
        df: dict[str, int] = {}
        by_len: dict[int, list[str]] = {}
        for term, doc in self._conn().execute("SELECT term, doc FROM fts_terms_vocab"):
            df[term] = int(doc)
            by_len.setdefault(len(term), []).append(term)
        self._vocab_df, self._vocab_by_len = df, by_len

    @property
    def vocab_df(self) -> dict[str, int]:
        if self._vocab_df is None:
            with self._vocab_lock:
                if self._vocab_df is None:
                    self._load_vocab()
        return self._vocab_df  # type: ignore[return-value]

    def vocab_near_length(self, n: int, spread: int = 2) -> list[str]:
        _ = self.vocab_df
        out: list[str] = []
        for k in range(max(1, n - spread), n + spread + 1):
            out.extend(self._vocab_by_len.get(k, []))  # type: ignore[union-attr]
        return out

    def descriptions_by_rowids(self, rowids: list[int]) -> dict[int, Description]:
        if not rowids:
            return {}
        out: dict[int, Description] = {}
        c = self._conn()
        for i in range(0, len(rowids), 500):
            chunk = rowids[i : i + 500]
            for r in c.execute(
                f"SELECT {_DESC_COLS} FROM search_terms WHERE rowid IN ({','.join('?' * len(chunk))})", chunk
            ):
                out[r[0]] = _desc(r)
        return out

    def descriptions_for_concept(self, concept_id: str) -> list[Description]:
        rows = self._conn().execute(f"SELECT {_DESC_COLS} FROM search_terms WHERE concept_id = ?", (concept_id,))
        return [_desc(r) for r in rows]
