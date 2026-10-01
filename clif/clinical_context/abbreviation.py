"""Context-sensitive abbreviation expansion.

An abbreviation is only resolved when one expansion dominates after combining
its prior with cue words found in the phrase and the surrounding context.
Otherwise all plausible expansions are kept and the mapping is flagged as
ambiguous, so the service never guesses between e.g. "multiple sclerosis" and
"mitral stenosis".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from clif.text import norm_fold


@dataclass(frozen=True)
class Expansion:
    text: str
    prior: float
    cues: tuple[str, ...] = ()
    language: str | None = None


@dataclass
class AbbreviationHit:
    abbreviation: str
    start: int
    end: int
    posteriors: list[tuple[str, float]]  # sorted, best first
    resolved: str | None
    cue_hits: dict[str, list[str]] = field(default_factory=dict)

    @property
    def plausible(self) -> list[tuple[str, float]]:
        return [(e, p) for e, p in self.posteriors if p >= 0.1]


class AbbreviationResolver:
    def __init__(self, table: dict[str, list[Expansion]], dominance: float = 0.9, cue_factor: float = 3.0):
        self.table = {k.upper(): v for k, v in table.items()}
        self.dominance = dominance
        self.cue_factor = cue_factor  # each supporting cue multiplies the odds
        self._cue_res: dict[str, re.Pattern] = {}

    @classmethod
    def from_yaml(cls, path: Path | None = None, dominance: float = 0.9) -> "AbbreviationResolver":
        if path is None:
            raw = yaml.safe_load(resources.files("clif.resources").joinpath("abbreviations.yaml").read_text("utf-8"))
        else:
            raw = yaml.safe_load(Path(path).read_text("utf-8"))
        table = {
            abbr: [
                Expansion(e["expansion"], float(e.get("prior", 1.0)), tuple(e.get("cues", []) or ()), e.get("language"))
                for e in entries
            ]
            for abbr, entries in (raw or {}).items()
        }
        return cls(table, dominance)

    def _cue_re(self, cue: str) -> re.Pattern:
        r = self._cue_res.get(cue)
        if r is None:
            r = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, norm_fold(cue).split())))
            self._cue_res[cue] = r
        return r

    def find(self, text: str, context: str | None) -> list[AbbreviationHit]:
        hits: list[AbbreviationHit] = []
        stripped = text.strip()
        evidence = norm_fold(f"{text} {context or ''}")
        for m in re.finditer(r"[A-Za-zÄÖÜäöü][A-Za-z0-9ÄÖÜäöü]{1,6}", text):
            tok = m.group(0)
            key = tok.upper()
            if key not in self.table:
                continue
            # Lower-case tokens only count when they are the entire input
            # ("ms" inside prose is more likely milliseconds than a diagnosis).
            if tok != key and tok.upper() != stripped.upper():
                continue
            expansions = self.table[key]
            scores: list[tuple[str, float]] = []
            cue_hits: dict[str, list[str]] = {}
            # Evidence must come from outside the abbreviation itself.
            ev = evidence.replace(norm_fold(tok), " ", 1)
            for e in expansions:
                found = [c for c in e.cues if self._cue_re(c).search(ev)]
                cue_hits[e.text] = found
                scores.append((e.text, e.prior * self.cue_factor ** len(found)))
            total = sum(s for _, s in scores) or 1.0
            post = sorted(((t, s / total) for t, s in scores), key=lambda x: -x[1])
            resolved = post[0][0] if post[0][1] >= self.dominance else None
            hits.append(AbbreviationHit(tok, m.start(), m.end(), post, resolved, cue_hits))
        return hits
