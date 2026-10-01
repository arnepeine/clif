"""Assertion attributes that are reported separately from the SNOMED concept."""

from __future__ import annotations

from enum import Enum


class Assertion(str, Enum):
    ASSERTED = "ASSERTED"
    NEGATED = "NEGATED"


class Certainty(str, Enum):
    CERTAIN = "CERTAIN"
    PROBABLE = "PROBABLE"
    POSSIBLE = "POSSIBLE"


class Experiencer(str, Enum):
    PATIENT = "PATIENT"
    FAMILY = "FAMILY"


# cue category (as produced by the analyzer) -> (attribute, value)
CUE_EFFECTS: dict[str, tuple[str, Enum]] = {
    "negation": ("assertion", Assertion.NEGATED),
    "certainty.possible": ("certainty", Certainty.POSSIBLE),
    "certainty.probable": ("certainty", Certainty.PROBABLE),
    "experiencer_family": ("experiencer", Experiencer.FAMILY),
}
