import pytest

from clif.clinical_context.analyzer import ContextAnalyzer
from clif.clinical_context.language import detect_language


@pytest.fixture(scope="module")
def analyzer():
    return ContextAnalyzer.default()


@pytest.mark.parametrize("text,core,assertion,temporality,certainty", [
    ("no chest pain", "chest pain", "NEGATED", "CURRENT", "CERTAIN"),
    ("patient denies chest pain", "chest pain", "NEGATED", "CURRENT", "CERTAIN"),
    ("history of myocardial infarction", "myocardial infarction", "ASSERTED", "HISTORICAL", "CERTAIN"),
    ("possible pneumonia", "pneumonia", "ASSERTED", "CURRENT", "POSSIBLE"),
    ("pneumonia?", "pneumonia", "ASSERTED", "CURRENT", "POSSIBLE"),
    ("probable pneumonia", "pneumonia", "ASSERTED", "CURRENT", "PROBABLE"),
    ("pulmonary embolism was ruled out", "pulmonary embolism", "NEGATED", "CURRENT", "CERTAIN"),
    ("kein Fieber", "Fieber", "NEGATED", "CURRENT", "CERTAIN"),
    ("Z.n. Appendektomie", "Appendektomie", "ASSERTED", "HISTORICAL", "CERTAIN"),
    ("V.a. Lungenembolie", "Lungenembolie", "ASSERTED", "CURRENT", "POSSIBLE"),
    ("Lungenembolie ausgeschlossen", "Lungenembolie", "NEGATED", "CURRENT", "CERTAIN"),
    ("planned appendectomy", "appendectomy", "ASSERTED", "PLANNED", "CERTAIN"),
    ("patient has severe pain in the right lower abdomen", "severe pain in the right lower abdomen",
     "ASSERTED", "CURRENT", "CERTAIN"),
    ("acute appendicitis", "acute appendicitis", "ASSERTED", "CURRENT", "CERTAIN"),
])
def test_assertion_split(analyzer, text, core, assertion, temporality, certainty):
    a = analyzer.analyze(text)
    assert a.core == core
    assert (a.assertion.value, a.temporality.value, a.certainty.value) == (assertion, temporality, certainty)
    assert a.original == text  # never modified


def test_family_history(analyzer):
    a = analyzer.analyze("family history of diabetes mellitus")
    assert a.core == "diabetes mellitus" and a.experiencer.value == "FAMILY"


def test_fixed_expression_not_split(analyzer):
    a = analyzer.analyze("no known drug allergies")
    assert a.fixed_expression and a.assertion.value == "ASSERTED" and a.core == "no known drug allergies"


def test_fixed_expression_disabled():
    a = ContextAnalyzer.default(situation_mode="off").analyze("no known drug allergies")
    assert not a.fixed_expression and a.assertion.value == "NEGATED" and a.core == "drug allergies"


def test_words_containing_cues_are_not_cues(analyzer):
    a = analyzer.analyze("nothing abnormal")  # "no" must match on word boundaries only
    assert a.assertion.value == "ASSERTED"


def test_insufficient(analyzer):
    assert analyzer.analyze("history of").insufficient
    assert analyzer.analyze("no").insufficient
    assert not analyzer.analyze("pain").insufficient


def test_multiple_clauses_warning(analyzer):
    a = analyzer.analyze("no fever but cough")
    assert a.core == "fever" and a.assertion.value == "NEGATED"
    assert any("multiple_clauses" in w for w in a.warnings)


@pytest.mark.parametrize("text,lang", [
    ("Schmerzen im rechten Unterbauch", "de"),
    ("patient has severe chest pain", "en"),
    ("douleur thoracique avec fièvre", "fr"),
])
def test_language_detection(text, lang):
    assert detect_language(text)[0] == lang


def test_soft_token_match_spelling_variants_and_suffixes():
    from clif.text import soft_token_match

    assert soft_token_match("anaemia", "anemia")
    assert soft_token_match("apendicitis", "appendicitis")
    assert not soft_token_match("appendicitis", "appendicectomy")
