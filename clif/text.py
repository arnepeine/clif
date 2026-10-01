"""Text normalization. The original text is never modified; we derive forms.

- ``norm``: NFKC, casefolded, punctuation -> space, whitespace collapsed.
- ``folded``: ``norm`` with German transliteration (ä->ae, ß->ss) and all
  remaining diacritics removed. Used for accent-insensitive matching.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SEMTAG_RE = re.compile(r"\s*\(([^()]+)\)\s*$")

_TRANSLIT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "æ": "ae", "œ": "oe", "ø": "o"})

# Function words removed for matching. Negation words are deliberately NOT
# here: "no known allergies" must keep "no" when mapped as a fixed expression.
STOPWORDS = frozenset(
    """
    a an the of in on at to for from by with and or is are was were be been being
    has have had this that these those as into its it his her their patient pt
    der die das des dem den ein eine einer eines einem einen und oder im in am an
    auf mit von vom zum zur bei ist sind war hat patientin patient
    de het een van en in op met voor bij is
    le la les un une des du et en au aux
    """.split()
)


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


@lru_cache(maxsize=65536)
def normalize(text: str) -> str:
    t = nfkc(text).casefold()
    t = t.replace("_", " ")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


@lru_cache(maxsize=65536)
def fold(text: str) -> str:
    """Accent-insensitive comparison form of an already normalized string."""
    t = text.translate(_TRANSLIT)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t


def norm_fold(text: str) -> str:
    return fold(normalize(text))


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def content_tokens(folded_text: str) -> list[str]:
    return [t for t in tokens(folded_text) if t not in STOPWORDS]


@lru_cache(maxsize=65536)
def stem(token: str) -> str:
    """Very light, language-agnostic suffix stripping for soft token matching.

    Deliberately conservative: it only needs to make "allergies"~"allergy" and
    "fibrillation"~"fibrillations" collide. Anything fancier belongs in the
    reranking model.
    """
    t = token
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    for suf in ("ungen", "ung", "es", "en", "s", "e"):
        if len(t) - len(suf) >= 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def strip_semantic_tag(fsn: str) -> tuple[str, str | None]:
    """'Appendicitis (disorder)' -> ('Appendicitis', 'disorder')."""
    m = _SEMTAG_RE.search(fsn)
    if not m:
        return fsn.strip(), None
    return fsn[: m.start()].strip(), m.group(1).strip()


@dataclass(frozen=True)
class QueryText:
    """A retrieval query derived from (part of) the input."""

    text: str
    norm: str
    folded: str
    source: str = "input"  # input | core | abbreviation | fixed_expression
    weight: float = 1.0
    content: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def build(cls, text: str, source: str = "input", weight: float = 1.0) -> "QueryText":
        n = normalize(text)
        f = fold(n)
        return cls(text=text, norm=n, folded=f, source=source, weight=weight, content=tuple(content_tokens(f)))


# Meaning-bearing medical suffixes: words that differ in these are different
# concepts even when they share a long root (appendic-itis vs appendic-ectomy).
_MEDICAL_SUFFIXES = (
    "ectomy", "ektomie", "otomy", "otomie", "ostomy", "ostomie", "plasty", "plastik", "scopy", "skopie",
    "graphy", "graphie", "itis", "itiden", "osis", "ose", "oma", "omas", "emia", "aemia", "ämie", "algia",
    "pathy", "pathie", "plegia", "cele", "rrhea", "rrhoea", "trophy",
)


def _medical_suffix(token: str) -> str | None:
    for suf in _MEDICAL_SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf) + 2:
            return suf
    return None


def soft_token_match(a: str, b: str) -> bool:
    """Tolerant token equality: identical, same light stem, same root with a
    short inflectional ending ("abdomen"~"abdominal"), or a small typo. Never
    equates words with different meaning-bearing suffixes."""
    if a == b or stem(a) == stem(b):
        return True
    if len(a) < 5 or len(b) < 5:
        return False
    sa, sb = _medical_suffix(a), _medical_suffix(b)
    if sa and sb and sa != sb:
        return False
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    if n >= 5 and max(len(a), len(b)) - n <= 4:
        return True
    from rapidfuzz.distance import Levenshtein

    limit = 1 if min(len(a), len(b)) < 8 else 2
    return Levenshtein.distance(a, b, score_cutoff=limit) <= limit


def coverage(query_tokens: list[str] | tuple[str, ...], target_tokens: list[str] | tuple[str, ...]) -> float:
    """Fraction of query tokens softly matched by some target token."""
    if not query_tokens:
        return 0.0
    hit = sum(1 for q in query_tokens if any(soft_token_match(q, t) for t in target_tokens))
    return hit / len(query_tokens)
