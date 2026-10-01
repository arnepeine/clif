"""Temporality attribute reported separately from the SNOMED concept."""

from __future__ import annotations

from enum import Enum


class Temporality(str, Enum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    PLANNED = "PLANNED"


CUE_EFFECTS: dict[str, tuple[str, Enum]] = {
    "temporality.historical": ("temporality", Temporality.HISTORICAL),
    "temporality.planned": ("temporality", Temporality.PLANNED),
}
