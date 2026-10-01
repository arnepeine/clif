"""Integration tests: retrieval -> reranking -> validation -> decision."""

import pytest

from clif.models.schemas import MapRequest
from clif.reranking.base import ConceptReranker, RerankRequest, RerankResult, result_from_label_scores


@pytest.mark.parametrize("text,concept_id", [
    ("acute appendicitis", "85189001"),
    ("appendicitis", "74400008"),
    ("Vorhofflimmern", "49436004"),
    ("appendectomy", "80146002"),
    ("no known drug allergies", "409137002"),
    ("history of myocardial infarction", "22298006"),
    ("patient denies chest pain", "29857009"),
    ("patient has severe pain in the right lower abdomen", "301754002"),
    ("heart attack", "22298006"),
    ("Herzinfarkt", "22298006"),
    ("fibrillation auriculaire", "49436004"),
])
def test_examples_from_brief(map_text, text, concept_id):
    r = map_text(text)
    assert r.status == "MATCH", (text, r.reason, [a.preferred_term for a in r.alternatives])
    assert r.concept.concept_id == concept_id
    assert r.snomed_version == "20250101"
    assert r.input == text


def test_output_structure(map_text):
    r = map_text("acute appendicitis")
    assert r.concept.preferred_term == "Acute appendicitis"
    assert r.concept.fsn == "Acute appendicitis (disorder)"
    assert r.concept.semantic_tag == "disorder"
    assert 0 < r.confidence <= 1
    assert r.alternatives and all(a.concept_id != "85189001" for a in r.alternatives)
    assert r.source_language == "en" and r.display_language == "en"
    assert r.reranker.backend == "heuristic"


def test_specificity_no_unsupported_detail(map_text):
    # "appendicitis" must not become "Acute appendicitis"
    assert map_text("appendicitis").concept.concept_id == "74400008"
    assert map_text("myocardial infarction").concept.concept_id == "22298006"
    assert map_text("atrial fibrillation").concept.concept_id == "49436004"


def test_negation_and_temporality_are_separate(map_text):
    r = map_text("no chest pain")
    assert (r.concept.concept_id, r.assertion) == ("29857009", "NEGATED")
    r = map_text("history of myocardial infarction")
    assert (r.concept.concept_id, r.temporality) == ("22298006", "HISTORICAL")
    # the situation concept must not be offered when context was split off
    assert all(a.concept_id != "399211009" for a in r.alternatives)
    r = map_text("possible pneumonia")
    assert (r.concept.concept_id, r.certainty) == ("233604007", "POSSIBLE")


def test_no_match(map_text):
    for t in ("the weather is lovely today", "xyzzy qwerty", "parking garage level three"):
        r = map_text(t)
        assert r.status == "NO_MATCH", (t, r.concept)
        assert r.concept is None and r.confidence == 0.0


def test_insufficient_information(map_text):
    for t in ("history of", "no", "patient", "?"):
        assert map_text(t).status == "INSUFFICIENT_INFORMATION", t


def test_ambiguous_close_competitors(map_text):
    r = map_text("diabetes")
    assert r.status == "AMBIGUOUS" and r.concept is None
    assert {"73211009", "15771004"} <= {a.concept_id for a in r.alternatives}


def test_hierarchy_restriction(map_text):
    assert map_text("appendectomy", allowed_hierarchies=["procedure"]).concept.concept_id == "80146002"
    r = map_text("appendicitis", allowed_hierarchies=["procedure"])
    assert r.status != "MATCH" or r.concept.semantic_tag == "procedure"
    assert map_text("appendicitis", allowed_hierarchies=["body structure"]).status in ("NO_MATCH", "AMBIGUOUS")


def test_ecl_restriction(mapper, map_text):
    r = map_text("diabetes", ecl="<< 64572001 |Disease|", debug=True)
    ids = [c["concept_id"] for c in r.debug["retrieval"]["candidates"]]
    assert ids and all(mapper.store.is_descendant_or_self(i, "64572001") for i in ids)


def test_context_disambiguates_but_does_not_inject(map_text):
    # context mentions pneumonia, but the phrase is about chest pain
    r = map_text("chest pain", context="Known pneumonia, on antibiotics.")
    assert r.concept.concept_id == "29857009"


def test_display_language(map_text):
    r = map_text("appendectomy", display_language="de")
    assert r.concept.preferred_term == "Appendektomie"


def test_debug_output(map_text):
    r = map_text("apendicitis", debug=True)
    d = r.debug
    for key in ("normalized_query", "retrieval", "reranker", "decision", "validation", "selected", "timing_ms"):
        assert key in d, key
    assert d["validation"]["passed"] is True
    assert d["retrieval"]["candidates"][0]["matched_descriptions"]
    assert map_text("apendicitis").debug is None


class _RogueReranker(ConceptReranker):
    """Returns an SCTID that was never offered, plus nonsense labels."""

    name = "rogue"

    def rerank(self, request: RerankRequest) -> RerankResult:
        return result_from_label_scores(request, {"999999999": 0.9, "ZZ": 0.05, request.candidates[0].label: 0.05},
                                        self.name)


def test_reranker_cannot_invent_sctid(mapper):
    from clif.pipeline import ClifMapper

    m = ClifMapper(mapper.settings, mapper.store, mapper.retriever, _RogueReranker(), mapper.analyzer,
                   mapper.policy)
    r = m.map(MapRequest(text="appendicitis", debug=True))
    assert r.concept is None or r.concept.concept_id != "999999999"
    assert "999999999" in r.debug["reranker"]["rejected"]
    returned = {a.concept_id for a in r.alternatives} | ({r.concept.concept_id} if r.concept else set())
    offered = {c["concept_id"] for c in r.debug["retrieval"]["candidates"]}
    assert returned <= offered


def test_batch(mapper):
    reqs = [MapRequest(text=t) for t in ("acute appendicitis", "Vorhofflimmern", "xyzzy qwerty", "no chest pain")]
    out = mapper.map_batch(reqs)
    assert [o.status for o in out] == ["MATCH", "MATCH", "NO_MATCH", "MATCH"]
    assert out[3].assertion == "NEGATED"


def test_batch_reports_item_errors(mapper):
    out = mapper.map_batch([MapRequest(text="appendicitis", ecl="<< banana"), MapRequest(text="appendicitis")])
    assert out[0].status == "ERROR" and out[1].status == "MATCH"


def test_decide_with_single_candidate():
    from clif.decision import DecisionPolicy, decide

    d = decide(DecisionPolicy(), {"74400008": 0.9}, 0.05, 0.05, {"74400008": 0.8})
    assert d.status == "MATCH" and d.margin == 0.9
