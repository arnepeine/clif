"""Global Patient Set flat-file format (ConceptID, Active, FSN, USPreferredTerm)."""

import zipfile

import pytest

from clif.config import Settings
from clif.indexing.build import build_index
from clif.models.schemas import MapRequest
from clif.pipeline import ClifMapper
from clif.reranking import HeuristicReranker
from clif.terminology.hierarchy import EclError
from clif.terminology.rf2 import import_packages

ROWS = [
    ("74400008", "1", "Appendicitis (disorder)", "Appendicitis"),
    ("85189001", "1", "Acute appendicitis (disorder)", "Acute appendicitis"),
    ("80146002", "1", "Appendectomy (procedure)", "Appendectomy"),
    ("29857009", "1", "Chest pain (finding)", "Chest pain"),
    ("196781001", "0", "Chronic appendicitis (disorder)", "Chronic appendicitis"),
]


@pytest.fixture(scope="module")
def gps_mapper(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gps")
    name = "SnomedINTL_GPSRelease_PRODUCTION_20260101T120000Z"
    zpath = tmp / f"{name}.zip"
    body = "ConceptID\tActive\tFSN\tUSPreferredTerm\r\n" + "".join("\t".join(r) + "\r\n" for r in ROWS)
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(f"{name}.txt", body)
        z.writestr("Readme.txt", "readme")
    s = Settings(data_dir=tmp / "data", _env_file=None)
    stats = import_packages([zpath], s.db_path, reset=True)
    assert stats.snomed_version == "20260101" and stats.rows["concept"] == 5
    build_index(s)
    return ClifMapper.from_settings(s, reranker=HeuristicReranker())


def test_gps_mapping(gps_mapper):
    r = gps_mapper.map(MapRequest(text="appendicitis"))
    assert r.status == "MATCH" and r.concept.concept_id == "74400008"
    assert r.concept.fsn == "Appendicitis (disorder)" and r.snomed_version == "20260101"
    assert not gps_mapper.store.has_hierarchy


def test_gps_inactive_excluded(gps_mapper):
    assert gps_mapper.map(MapRequest(text="chronic appendicitis")).concept.concept_id != "196781001"


def test_gps_hierarchy_falls_back_to_semantic_tag(gps_mapper):
    r = gps_mapper.map(MapRequest(text="appendix", allowed_hierarchies=["procedure"], debug=True))
    assert all(c["semantic_tag"] == "procedure" for c in r.debug["retrieval"]["candidates"])
    with pytest.raises(EclError, match="IS-A relationships"):
        gps_mapper.map(MapRequest(text="appendicitis", ecl="<< 64572001"))
