"""Official SNOMED CT RF2 release importer.

Loads concepts, descriptions, (inferred) relationships and language reference
sets from one or more RF2 packages (directories or .zip files) into SQLite.

Rows are upserted by component id, keeping the row with the latest
effectiveTime. This one rule makes Snapshot, Full and Delta files as well as
layered extensions (International + national edition) behave correctly.
Inactive rows are kept in the database; filtering to active content happens
when the search index is built and at query time.

Usage:
    python -m clif.import_rf2 /data/SnomedCT_InternationalRF2_PRODUCTION_20250101T120000Z
"""

from __future__ import annotations

import io
import logging
import re
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from clif.terminology.sctid import is_valid_sctid

log = logging.getLogger(__name__)

FSN_TYPE = "900000000000003001"
SYNONYM_TYPE = "900000000000013009"
IS_A = "116680003"
PREFERRED = "900000000000548007"
ACCEPTABLE = "900000000000549004"
INFERRED = "900000000000011006"

SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts(
    id TEXT PRIMARY KEY, effective_time TEXT NOT NULL, active INTEGER NOT NULL,
    module_id TEXT, definition_status_id TEXT) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS descriptions(
    id TEXT PRIMARY KEY, effective_time TEXT NOT NULL, active INTEGER NOT NULL,
    module_id TEXT, concept_id TEXT NOT NULL, language_code TEXT NOT NULL,
    type_id TEXT NOT NULL, term TEXT NOT NULL, case_significance_id TEXT) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS relationships(
    id TEXT PRIMARY KEY, effective_time TEXT NOT NULL, active INTEGER NOT NULL,
    module_id TEXT, source_id TEXT NOT NULL, destination_id TEXT NOT NULL,
    relationship_group INTEGER, type_id TEXT NOT NULL,
    characteristic_type_id TEXT, modifier_id TEXT) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS langrefset(
    id TEXT PRIMARY KEY, effective_time TEXT NOT NULL, active INTEGER NOT NULL,
    module_id TEXT, refset_id TEXT NOT NULL, referenced_component_id TEXT NOT NULL,
    acceptability_id TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS release_packages(
    name TEXT, path TEXT, release_type TEXT, release_date TEXT, imported_at TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS ix_desc_concept ON descriptions(concept_id);
CREATE INDEX IF NOT EXISTS ix_rel_source ON relationships(source_id, type_id);
CREATE INDEX IF NOT EXISTS ix_rel_dest ON relationships(destination_id, type_id);
CREATE INDEX IF NOT EXISTS ix_lang_component ON langrefset(referenced_component_id);
"""

_COLUMNS = {
    "concept": ("concepts", ["id", "effective_time", "active", "module_id", "definition_status_id"]),
    "description": (
        "descriptions",
        ["id", "effective_time", "active", "module_id", "concept_id", "language_code", "type_id", "term",
         "case_significance_id"],
    ),
    "relationship": (
        "relationships",
        ["id", "effective_time", "active", "module_id", "source_id", "destination_id", "relationship_group",
         "type_id", "characteristic_type_id", "modifier_id"],
    ),
    "language": (
        "langrefset",
        ["id", "effective_time", "active", "module_id", "refset_id", "referenced_component_id",
         "acceptability_id"],
    ),
}

_FILE_PATTERNS = {
    "concept": re.compile(r"sct2_Concept_(Snapshot|Full|Delta)[-_]", re.I),
    "description": re.compile(r"sct2_Description_(Snapshot|Full|Delta)[-_]", re.I),
    "relationship": re.compile(r"sct2_Relationship_(Snapshot|Full|Delta)[-_]", re.I),
    "language": re.compile(r"der2_cRefset_Language(Snapshot|Full|Delta)[-_]", re.I),
}
_RELEASE_DATE_RE = re.compile(r"(\d{8})T\d{6}Z")
# SNOMED International Global Patient Set: one flat file (ConceptID, Active, FSN, USPreferredTerm)
_GPS_FILE_RE = re.compile(r"GPSRelease.*\.txt$", re.I)
GPS_HEADER = ["ConceptID", "Active", "FSN", "USPreferredTerm"]


@dataclass
class Rf2File:
    kind: str
    release_type: str
    name: str
    opener: object  # callable returning a text stream


@dataclass
class ImportStats:
    rows: dict
    invalid_ids: int
    seconds: float
    snomed_version: str


def _discover(path: Path) -> Iterator[Rf2File]:
    def classify(name: str):
        base = name.rsplit("/", 1)[-1]
        if not base.endswith(".txt"):
            return None
        if _GPS_FILE_RE.search(base):
            return "gps", "snapshot"
        for kind, pat in _FILE_PATTERNS.items():
            m = pat.search(base)
            if m:
                return kind, m.group(1).lower()
        return None

    if path.is_file() and path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        for info in zf.infolist():
            c = classify(info.filename)
            if c:
                yield Rf2File(c[0], c[1], info.filename,
                              lambda n=info.filename: io.TextIOWrapper(zf.open(n), encoding="utf-8", newline=""))
    elif path.is_dir():
        for p in sorted(path.rglob("*.txt")):
            c = classify(p.name)
            if c:
                yield Rf2File(c[0], c[1], str(p), lambda p=p: open(p, encoding="utf-8", newline=""))
    else:
        raise FileNotFoundError(f"RF2 package not found or not a directory/zip: {path}")


def _select(files: list[Rf2File], release_type: str) -> list[Rf2File]:
    """Per kind, use the requested release type; fall back to Snapshot, then Full."""
    chosen: list[Rf2File] = [f for f in files if f.kind == "gps"]
    order = [release_type] + [t for t in ("snapshot", "full") if t != release_type]
    for kind in _COLUMNS:
        of_kind = [f for f in files if f.kind == kind]
        for rt in order:
            sel = [f for f in of_kind if f.release_type == rt]
            if sel:
                chosen.extend(sel)
                break
    return chosen


def _rows(stream, ncols: int) -> Iterator[list[str]]:
    with stream as fh:
        header = fh.readline()
        if not header:
            return
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != ncols:
                raise ValueError(f"Malformed RF2 row ({len(parts)} columns, expected {ncols}): {line[:120]!r}")
            yield parts


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def import_packages(
    paths: Iterable[Path],
    db_path: Path,
    release_type: str = "snapshot",
    reset: bool = False,
    validate_ids: bool = False,
    batch_size: int = 20000,
) -> ImportStats:
    t0 = time.time()
    if reset and db_path.exists():
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    conn = connect(db_path)
    counts: dict[str, int] = {k: 0 for k in _COLUMNS}
    invalid = 0
    release_dates: list[str] = []

    for path in paths:
        path = Path(path)
        files = _select(list(_discover(path)), release_type)
        if not files:
            raise ValueError(f"No RF2 concept/description/relationship/language files found in {path}")
        m = _RELEASE_DATE_RE.search(path.name)
        rdate = m.group(1) if m else None
        if rdate:
            release_dates.append(rdate)
        log.info("Importing %s (%d files)", path, len(files))
        for f in files:
            if f.kind == "gps":
                n = _import_gps(conn, f, rdate or "00000000", batch_size)
                counts["concept"] += n
                counts["description"] += 2 * n
                log.info("  gps          %8d concepts  %s", n, f.name.rsplit("/", 1)[-1])
                continue
            table, cols = _COLUMNS[f.kind]
            placeholders = ",".join("?" * len(cols))
            updates = ",".join(f"{c}=excluded.{c}" for c in cols[1:])
            sql = (
                f"INSERT INTO {table}({','.join(cols)}) VALUES({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates} WHERE excluded.effective_time >= {table}.effective_time"
            )
            batch: list[list] = []
            n = 0
            for parts in _rows(f.opener(), len(cols)):
                if f.kind == "relationship" and parts[8] != INFERRED:
                    continue
                parts[2] = 1 if parts[2] == "1" else 0
                if validate_ids and not is_valid_sctid(parts[0]):
                    invalid += 1
                batch.append(parts)
                if len(batch) >= batch_size:
                    conn.executemany(sql, batch)
                    n += len(batch)
                    batch.clear()
            if batch:
                conn.executemany(sql, batch)
                n += len(batch)
            counts[f.kind] += n
            log.info("  %-12s %-9s %8d rows  %s", f.kind, f.release_type, n, f.name.rsplit("/", 1)[-1])
        conn.execute(
            "INSERT INTO release_packages VALUES(?,?,?,?,datetime('now'))",
            (path.name, str(path), release_type, rdate),
        )
        conn.commit()

    conn.executescript(INDEXES)
    # Version: the release date from the package name if present, otherwise the
    # latest effectiveTime of any imported concept or description.
    max_et = conn.execute(
        "SELECT max(m) FROM (SELECT max(effective_time) m FROM concepts UNION ALL "
        "SELECT max(effective_time) FROM descriptions)"
    ).fetchone()[0]
    all_dates = [r[0] for r in conn.execute("SELECT release_date FROM release_packages") if r[0]]
    version = max(all_dates) if all_dates else (max_et or "unknown")
    editions = [r[0] for r in conn.execute("SELECT name FROM release_packages ORDER BY rowid")]
    conn.executemany(
        "INSERT OR REPLACE INTO meta VALUES(?,?)",
        [("snomed_version", version), ("max_effective_time", max_et or ""), ("snomed_edition", " + ".join(editions))],
    )
    conn.commit()
    conn.execute("ANALYZE")
    conn.close()
    return ImportStats(rows=counts, invalid_ids=invalid, seconds=time.time() - t0, snomed_version=version)


def _import_gps(conn: sqlite3.Connection, f: Rf2File, effective_time: str, batch_size: int) -> int:
    """Global Patient Set: each row becomes a concept plus an FSN and a preferred-term
    description. The GPS has no description IDs, so deterministic synthetic ones are
    used (``gps:<conceptId>:fsn|pt``); they are never returned as SCTIDs."""
    concepts, descs = [], []
    n = 0
    csql = ("INSERT INTO concepts VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET effective_time=excluded."
            "effective_time, active=excluded.active WHERE excluded.effective_time >= concepts.effective_time")
    dsql = ("INSERT INTO descriptions VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET term=excluded.term, "
            "active=excluded.active, effective_time=excluded.effective_time "
            "WHERE excluded.effective_time >= descriptions.effective_time")
    with f.opener() as fh:  # type: ignore[operator]
        header = fh.readline().rstrip("\r\n").split("\t")
        if header != GPS_HEADER:
            raise ValueError(f"unexpected GPS header {header}, expected {GPS_HEADER}")
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 4 or not parts[0]:
                continue
            cid, active, fsn, pt = parts
            act = 1 if active == "1" else 0
            concepts.append((cid, effective_time, act, None, None))
            descs.append((f"gps:{cid}:fsn", effective_time, act, None, cid, "en", FSN_TYPE, fsn, None))
            if pt:
                descs.append((f"gps:{cid}:pt", effective_time, act, None, cid, "en", SYNONYM_TYPE, pt, None))
            n += 1
            if len(concepts) >= batch_size:
                conn.executemany(csql, concepts)
                conn.executemany(dsql, descs)
                concepts.clear()
                descs.clear()
    if concepts:
        conn.executemany(csql, concepts)
        conn.executemany(dsql, descs)
    return n
