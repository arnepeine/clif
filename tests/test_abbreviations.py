from clif.clinical_context.abbreviation import AbbreviationResolver


def resolver():
    return AbbreviationResolver.from_yaml(dominance=0.9)


def test_unambiguous_abbreviation_resolves():
    (hit,) = resolver().find("COPD", None)
    assert hit.resolved == "chronic obstructive pulmonary disease"


def test_ambiguous_without_context():
    (hit,) = resolver().find("MS", None)
    assert hit.resolved is None
    assert {e for e, _ in hit.plausible} == {"multiple sclerosis", "mitral stenosis"}


def test_context_resolves():
    (hit,) = resolver().find("AF", "ECG demonstrates irregularly irregular rhythm. Previous diagnosis of atrial "
                                   "fibrillation.")
    assert hit.resolved == "atrial fibrillation"
    (hit,) = resolver().find("MS", "Echo shows thickened mitral valve leaflets and a diastolic murmur")
    assert hit.resolved == "mitral stenosis"
    (hit,) = resolver().find("RA", "SpO2 94% on RA, oxygen saturation stable")
    assert hit.resolved == "room air"


def test_single_weak_cue_does_not_resolve_three_way_abbreviation():
    (hit,) = resolver().find("RA", "SpO2 94%")
    assert hit.resolved is None


def test_lowercase_inside_prose_is_not_an_abbreviation():
    assert resolver().find("latency 5 ms", None) == []
    assert len(resolver().find("ms", None)) == 1  # whole input


def test_pipeline_abbreviation(map_text):
    assert map_text("COPD").concept.concept_id == "13645005"
    r = map_text("MS")
    assert r.status == "AMBIGUOUS" and r.reason.startswith("ambiguous_abbreviation")
    assert {a.concept_id for a in r.alternatives[:2]} == {"24700007", "79619009"}
    r = map_text("MS", context="MRI shows periventricular demyelinating lesions, relapsing course")
    assert r.status == "MATCH" and r.concept.concept_id == "24700007"


def test_af_with_context_example(map_text):
    r = map_text("AF", context="ECG demonstrates irregularly irregular rhythm. Previous diagnosis of atrial "
                               "fibrillation.")
    assert r.status == "MATCH" and r.concept.concept_id == "49436004"
    assert map_text("AF").status == "AMBIGUOUS"
