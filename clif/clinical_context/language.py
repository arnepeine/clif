"""Lightweight source-language detection for short clinical phrases.

Only used when the caller does not specify ``language``. Returns
(language, confident). When not confident, retrieval searches all indexed
languages, so a wrong guess cannot hide the right description.
"""

from __future__ import annotations

import re

from clif.text import normalize, tokens

_MARKERS: dict[str, set[str]] = {
    "de": {"der", "die", "das", "und", "mit", "nicht", "kein", "keine", "bei", "nach", "zustand", "verdacht",
           "auf", "ist", "hat", "eine", "einer", "schmerzen", "links", "rechts", "akute", "akuter", "chronische"},
    "nl": {"de", "het", "een", "van", "geen", "niet", "met", "bij", "na", "en", "is", "heeft", "pijn", "links",
           "rechts", "acute", "chronische"},
    "fr": {"le", "la", "les", "des", "du", "pas", "avec", "sans", "et", "une", "douleur", "aigue", "aiguë",
           "chronique", "gauche", "droite"},
    "en": {"the", "of", "and", "with", "no", "not", "without", "history", "pain", "left", "right", "acute",
           "chronic", "patient", "has", "denies", "severe"},
}
_SUFFIXES: dict[str, tuple[str, ...]] = {
    "de": ("ung", "ungen", "keit", "heit", "entzündung", "schmerz", "schmerzen", "flimmern", "infarkt", "insuffizienz",
           "ektomie", "tomie", "ose", "itis"),
    "nl": ("ing", "heid", "ontsteking", "pijn"),
    "fr": ("ite", "ique", "ement", "tion"),
    "en": ("ing", "tion", "ectomy", "otomy", "itis", "osis", "ness"),
}


def detect_language(text: str) -> tuple[str, bool]:
    n = normalize(text)
    toks = tokens(n)
    if not toks:
        return "en", False
    scores = {lang: 0.0 for lang in _MARKERS}
    if re.search(r"[äöüß]", n):
        scores["de"] += 2.0
    if re.search(r"[éèêàçëï]", n):
        scores["fr"] += 1.5
    if "ij" in n:
        scores["nl"] += 0.5
    for t in toks:
        for lang, words in _MARKERS.items():
            if t in words:
                scores[lang] += 1.0
        for lang, sufs in _SUFFIXES.items():
            if any(t.endswith(s) and len(t) > len(s) + 2 for s in sufs):
                scores[lang] += 0.5
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0] != "en"))
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 == 0:
        return "en", False
    return best, s1 >= 1.5 and s1 - s2 >= 1.0
