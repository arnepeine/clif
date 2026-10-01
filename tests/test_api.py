import pytest
from fastapi.testclient import TestClient

from clif.api import create_app


@pytest.fixture(scope="module")
def client(settings, mapper):
    with TestClient(create_app(settings, mapper=mapper)) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["snomed_version"] == "20250101" and h["reranker"]["backend"] == "heuristic"


def test_map(client):
    r = client.post("/map", json={"text": "acute appendicitis", "language": "en", "top_k": 3}).json()
    assert r["status"] == "MATCH"
    assert r["concept"]["concept_id"] == "85189001"
    assert r["concept"]["preferred_term"] == "Acute appendicitis"
    assert len(r["alternatives"]) <= 3
    assert r["debug"] is None
    assert r["snomed_version"] == "20250101"


def test_map_debug(client):
    r = client.post("/map", json={"text": "Herzinfarkt", "debug": True}).json()
    assert r["debug"]["retrieval"]["candidates"]
    assert r["matched_description_language"] == "de" and r["source_language"] == "de"


def test_map_batch(client):
    r = client.post("/map/batch", json={"items": [{"text": "appendectomy"}, {"text": "MS"}]}).json()
    assert [x["status"] for x in r["results"]] == ["MATCH", "AMBIGUOUS"]


def test_candidates(client):
    r = client.post("/candidates", json={"text": "Herzinfarkt", "language": "de", "limit": 32}).json()
    assert r["candidates"][0]["concept_id"] == "22298006"
    assert r["candidates"][0]["matched_descriptions"][0]["term"] == "Herzinfarkt"


def test_concept(client):
    r = client.get("/concepts/85189001").json()
    assert r["parents"] == ["74400008"] and "138875005" in r["ancestors"]
    assert client.get("/concepts/1234567").status_code == 404


def test_bad_requests(client):
    assert client.post("/map", json={"text": "x", "ecl": "<< 1 : 2 = 3"}).status_code == 400
    assert client.post("/map", json={"text": "x", "allowed_hierarchies": ["nope"]}).status_code == 400
    assert client.post("/map", json={"text": ""}).status_code == 422
