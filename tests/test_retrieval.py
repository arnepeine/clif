from clif.models.schemas import CandidatesRequest


def _cands(mapper, text, **kw):
    return mapper.candidates(CandidatesRequest(text=text, **kw))


def test_exact_preferred_term_ranks_first(mapper):
    r = _cands(mapper, "acute appendicitis")
    assert r.candidates[0].concept_id == "85189001"
    assert r.candidates[0].exact_or_synonym_bonus == 1.0


def test_synonym_match(mapper):
    r = _cands(mapper, "heart attack")
    assert r.candidates[0].concept_id == "22298006"
    assert r.candidates[0].matched_descriptions[0]["term"] == "Heart attack"


def test_fuzzy_typo(mapper):
    r = _cands(mapper, "apendicitis")
    assert r.candidates[0].concept_id == "74400008"
    assert "fuzzy" in r.candidates[0].matched_descriptions[0]["channels"]


def test_german_input(mapper):
    r = _cands(mapper, "Herzinfarkt", language="de")
    assert r.searched_languages == ["de", "en"]
    assert r.candidates[0].concept_id == "22298006"
    assert r.candidates[0].matched_descriptions[0]["language"] == "de"


def test_accent_folding(mapper):
    assert _cands(mapper, "Blinddarmentzundung").candidates[0].concept_id == "74400008"


def test_duplicate_descriptions_collapse_to_one_concept(mapper):
    # "Appendicitis" exists in en (US+GB membership) and nl; all map to one SCTID
    r = _cands(mapper, "appendicitis")
    ids = [c.concept_id for c in r.candidates]
    assert len(ids) == len(set(ids))
    top = r.candidates[0]
    assert top.concept_id == "74400008"
    assert len({m["language"] for m in top.matched_descriptions}) >= 2


def test_inactive_concepts_excluded(mapper):
    ids = [c.concept_id for c in _cands(mapper, "Appendicitis NOS", limit=100).candidates]
    from clif.devtools.demo_release import fake

    assert fake(2) not in ids


def test_candidate_limit_and_scores(mapper):
    r = _cands(mapper, "pain", limit=32)
    assert 0 < len(r.candidates) <= 32
    scores = [c.retrieval_score for c in r.candidates]
    assert scores == sorted(scores, reverse=True)


def test_hierarchy_restriction_in_retrieval(mapper):
    r = _cands(mapper, "appendix", allowed_hierarchies=["procedure"])
    assert r.candidates and all(c.semantic_tag == "procedure" for c in r.candidates)
    assert any(f["reason"].startswith("not in allowed hierarchies") for f in r.filtered)


def test_retrieval_cache(mapper):
    _cands(mapper, "chest pain")
    p = mapper.prepare("chest pain", None, "auto", None, None)
    assert mapper._retrieve(p).cached


def test_vector_channel_contributes(mapper):
    assert mapper.retriever.vector_enabled
    r = _cands(mapper, "atrial fibrillation")
    assert r.channel_counts.get("vector", 0) > 0
