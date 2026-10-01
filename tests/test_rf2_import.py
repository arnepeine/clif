import shutil
import sqlite3
import zipfile

from clif.config import Settings
from clif.indexing.build import build_index
from clif.terminology.rf2 import import_packages
from clif.terminology.sctid import is_valid_sctid, make_sctid
from clif.terminology.store import TerminologyStore


def test_import_counts_and_version(settings):
    conn = sqlite3.connect(settings.db_path)
    assert conn.execute("SELECT count(*) FROM concepts").fetchone()[0] > 100
    assert conn.execute("SELECT value FROM meta WHERE key='snomed_version'").fetchone()[0] == "20250101"
    # SCTIDs are stored as text, never numbers
    assert conn.execute("SELECT typeof(id) FROM concepts LIMIT 1").fetchone()[0] == "text"
    assert conn.execute("SELECT typeof(concept_id) FROM search_terms LIMIT 1").fetchone()[0] == "text"


def test_long_sctid_preserved_exactly(mapper):
    # 18-digit metadata IDs would lose precision as floats
    rec = mapper.store.concept("85189001")
    assert rec.concept_id == "85189001"
    assert all(isinstance(p, str) for p in rec.parents)


def test_inactive_concept_and_description_not_searchable(mapper):
    conn = mapper.store._conn()
    assert conn.execute("SELECT count(*) FROM search_terms WHERE term = 'Appendicitis NOS'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM search_terms WHERE term = 'Appendicitis unspecified'").fetchone()[0] == 0
    assert not mapper.store.is_active(conn.execute(
        "SELECT concept_id FROM descriptions WHERE term = 'Appendicitis NOS'").fetchone()[0])


def test_inactive_isa_edge_ignored(mapper):
    # 85189001 had an inactive IS-A to Disease; its only active parent is Appendicitis
    assert mapper.store.concept("85189001").parents == ["74400008"]
    assert "64572001" in mapper.store.ancestors("85189001")  # via Appendicitis


def test_preferred_terms_per_language(mapper):
    rec = mapper.store.concept("80146002")
    assert rec.preferred_terms["en"] == "Appendectomy"  # US refset is first in the configured order
    assert rec.preferred_terms["de"] == "Appendektomie"
    assert rec.preferred_term("fr")[1] == "en"  # no French PT -> English fallback


def test_gb_refset_first_changes_preferred_term(tmp_path, demo_package):
    s = Settings(data_dir=tmp_path, _env_file=None,
                 language_refsets={"en": ["900000000000508004", "900000000000509007"]})
    import_packages([demo_package], s.db_path, reset=True)
    build_index(s)
    assert TerminologyStore(s.db_path).concept("80146002").preferred_terms["en"] == "Appendicectomy"


def test_zip_import_and_delta_upsert(tmp_path, demo_package):
    zpath = tmp_path / (demo_package.name + ".zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for f in demo_package.rglob("*.txt"):
            z.write(f, f.relative_to(demo_package.parent))
    db = tmp_path / "t.sqlite"
    import_packages([zpath], db, reset=True)
    # A later Delta inactivates Chest pain: the newer row must win.
    delta = tmp_path / "SnomedCT_Delta_20250701T120000Z" / "Delta" / "Terminology"
    delta.mkdir(parents=True)
    (delta / "sct2_Concept_Delta_INT_20250701.txt").write_text(
        "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\r\n"
        "29857009\t20250701\t0\t900000000000207008\t900000000000074008\r\n")
    # An older row must NOT override the newer one.
    old = tmp_path / "SnomedCT_Old_20200101T120000Z" / "Delta" / "Terminology"
    old.mkdir(parents=True)
    (old / "sct2_Concept_Delta_INT_20200101.txt").write_text(
        "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\r\n"
        "29857009\t20200101\t1\t900000000000207008\t900000000000074008\r\n")
    stats = import_packages([delta.parent.parent, old.parent.parent], db, release_type="delta")
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT active, effective_time FROM concepts WHERE id='29857009'").fetchone() == (0, "20250701")
    assert stats.snomed_version == "20250701"
    shutil.rmtree(tmp_path / "SnomedCT_Old_20200101T120000Z")


def test_verhoeff():
    assert is_valid_sctid("85189001")
    assert is_valid_sctid("900000000000509007")
    assert not is_valid_sctid("85189002")
    assert is_valid_sctid(make_sctid(12345))


def test_stale_index_schema_gives_clear_error(tmp_path, settings):
    import pytest

    db = tmp_path / "old.sqlite"
    shutil.copy(settings.db_path, db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'index_schema'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="rerun `python -m clif.build_index`"):
        TerminologyStore(db)


def test_release_without_language_refsets_is_still_searchable(tmp_path, demo_package):
    """Slim packages (e.g. the Global Patient Set) may ship without language refsets."""
    pkg = tmp_path / demo_package.name
    shutil.copytree(demo_package, pkg)
    shutil.rmtree(pkg / "Snapshot" / "Refset")
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    import_packages([pkg], s.db_path, reset=True)
    build_index(s)
    store = TerminologyStore(s.db_path)
    assert store.exact("acute appendicitis", "acute appendicitis", ["en"])
    assert store.concept("85189001").preferred_terms["en"] == "Acute appendicitis"
