"""Hierarchy restrictions: named top-level hierarchies and an ECL subset.

Constraints are evaluated as predicates on candidate concepts (after
retrieval, before reranking, and again during validation), so even
``<< 404684003`` (hundreds of thousands of concepts) never has to be expanded.

Supported ECL (v2 brief syntax):
    focus:        SCTID [|term|]  or  *
    operators:    <<  <  <!  >>  >  >!   (and the same without operator = self)
    compound:     AND  OR  MINUS  ,(=AND)  with parentheses
Refinements (``:``), member-of (``^``), dotted attributes and filters raise
``EclUnsupported`` so callers get an explicit 400 instead of a silent wrong
result. The parser is intentionally a separate class so full ECL (e.g. via a
Snowstorm/Ontoserver-compatible evaluator) can be plugged in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

NAMED_HIERARCHIES: dict[str, str] = {
    "clinical finding": "404684003",
    "finding": "404684003",
    "disorder": "64572001",
    "disease": "64572001",
    "procedure": "71388002",
    "body structure": "123037004",
    "observable entity": "363787002",
    "situation": "243796009",
    "situation with explicit context": "243796009",
    "product": "373873005",
    "pharmaceutical / biologic product": "373873005",
    "substance": "105590001",
    "organism": "410607006",
    "specimen": "123038009",
    "physical object": "260787004",
    "event": "272379006",
    "qualifier value": "362981000",
    "social context": "48176007",
    "environment": "308916002",
    "environment or geographical location": "308916002",
    "record artifact": "419891008",
    "physical force": "78621006",
    "staging and scales": "254291000",
}


# Used when a release ships without relationships (e.g. the Global Patient Set):
# top-level hierarchy -> semantic tags whose concepts live in it.
HIERARCHY_TAGS: dict[str, set[str]] = {
    "404684003": {"finding", "disorder"},
    "64572001": {"disorder"},
    "71388002": {"procedure", "regime/therapy"},
    "123037004": {"body structure", "morphologic abnormality", "cell", "cell structure"},
    "363787002": {"observable entity"},
    "243796009": {"situation"},
    "373873005": {"product", "medicinal product", "medicinal product form", "clinical drug", "real clinical drug"},
    "105590001": {"substance"},
    "410607006": {"organism"},
    "123038009": {"specimen"},
    "260787004": {"physical object"},
    "272379006": {"event"},
    "362981000": {"qualifier value"},
    "48176007": {"social concept", "person", "occupation", "ethnic group", "racial group", "religion/philosophy",
                 "life style"},
    "308916002": {"environment", "geographic location", "environment / location"},
    "419891008": {"record artifact"},
    "78621006": {"physical force"},
    "254291000": {"staging scale", "assessment scale", "tumor staging"},
}


class HierarchySource(Protocol):
    def ancestors(self, concept_id: str) -> frozenset[str]: ...

    def concept(self, concept_id: str): ...


class EclError(ValueError):
    pass


class EclUnsupported(EclError):
    pass


# --- ECL AST -----------------------------------------------------------------


@dataclass(frozen=True)
class Focus:
    op: str  # "", "<<", "<", "<!", ">>", ">", ">!"
    concept_id: str  # or "*"

    def matches(self, cid: str, h: HierarchySource) -> bool:
        x = self.concept_id
        if x == "*":
            return self.op in ("", "<<", ">>") or cid != x
        if self.op == "":
            return cid == x
        if self.op == "<<":
            return cid == x or x in h.ancestors(cid)
        if self.op == "<":
            return x in h.ancestors(cid)
        if self.op == "<!":
            rec = h.concept(cid)
            return bool(rec and x in rec.parents)
        if self.op == ">>":
            return cid == x or cid in h.ancestors(x)
        if self.op == ">":
            return cid in h.ancestors(x)
        if self.op == ">!":
            rec = h.concept(x)
            return bool(rec and cid in rec.parents)
        raise EclError(f"unknown operator {self.op}")


@dataclass(frozen=True)
class Compound:
    op: str  # AND | OR | MINUS
    left: object
    right: object

    def matches(self, cid: str, h: HierarchySource) -> bool:
        a = self.left.matches(cid, h)  # type: ignore[attr-defined]
        if self.op == "AND":
            return a and self.right.matches(cid, h)  # type: ignore[attr-defined]
        if self.op == "OR":
            return a or self.right.matches(cid, h)  # type: ignore[attr-defined]
        return a and not self.right.matches(cid, h)  # type: ignore[attr-defined]


_TOKEN_RE = re.compile(
    r"\s*(?:(?P<op><<!|<<|<!|<|>>!|>>|>!|>)|(?P<id>\d{6,18})|(?P<star>\*)|(?P<term>\|[^|]*\|)"
    r"|(?P<kw>AND|OR|MINUS)\b|(?P<comma>,)|(?P<lp>\()|(?P<rp>\))|(?P<bad>.))",
    re.I,
)


class EclParser:
    def __init__(self, text: str):
        self.text = text
        self.toks: list[tuple[str, str]] = []
        for m in _TOKEN_RE.finditer(text):
            kind = m.lastgroup
            val = m.group(kind) if kind else ""
            if kind == "term":
                continue
            if kind == "bad":
                if val.strip() in (":", "^", ".", "{", "!", "["):
                    raise EclUnsupported(f"ECL feature '{val}' is not supported yet: {text!r}")
                if val.strip():
                    raise EclError(f"unexpected character {val!r} in ECL {text!r}")
                continue
            self.toks.append((kind or "", val.upper() if kind == "kw" else val))
        self.i = 0

    def parse(self):
        if not self.toks:
            raise EclError("empty ECL expression")
        node = self._expr()
        if self.i != len(self.toks):
            raise EclError(f"unexpected token {self.toks[self.i][1]!r} in ECL {self.text!r}")
        return node

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else ("", "")

    def _expr(self):
        node = self._sub()
        while True:
            kind, val = self._peek()
            if kind == "kw" or kind == "comma":
                self.i += 1
                op = "AND" if kind == "comma" else val
                node = Compound(op, node, self._sub())
            else:
                return node

    def _sub(self):
        kind, val = self._peek()
        if kind == "lp":
            self.i += 1
            node = self._expr()
            if self._peek()[0] != "rp":
                raise EclError("missing ')' in ECL")
            self.i += 1
            return node
        op = ""
        if kind == "op":
            if val in ("<<!", ">>!"):
                raise EclUnsupported(f"ECL operator {val} is not supported yet")
            op = val
            self.i += 1
            kind, val = self._peek()
            if kind == "lp":  # e.g. << (A OR B)
                raise EclUnsupported("constraint operators applied to nested expressions are not supported yet")
        if kind == "id":
            self.i += 1
            return Focus(op, val)
        if kind == "star":
            self.i += 1
            return Focus(op, "*")
        raise EclError(f"expected concept id in ECL {self.text!r}")


def parse_ecl(text: str):
    return EclParser(text).parse()


@dataclass
class HierarchyConstraint:
    """Caller-supplied restriction of admissible concepts."""

    allowed_roots: list[str] = field(default_factory=list)
    ecl_text: str | None = None
    ecl: object | None = None
    labels: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, allowed_hierarchies: list[str] | None, ecl: str | None) -> "HierarchyConstraint | None":
        if not allowed_hierarchies and not ecl:
            return None
        roots: list[str] = []
        for h in allowed_hierarchies or []:
            key = h.strip().lower()
            if key in NAMED_HIERARCHIES:
                roots.append(NAMED_HIERARCHIES[key])
            elif re.fullmatch(r"\d{6,18}", key):
                roots.append(key)
            else:
                raise ValueError(f"unknown hierarchy {h!r}; use a name from {sorted(NAMED_HIERARCHIES)} or an SCTID")
        return cls(allowed_roots=roots, ecl_text=ecl, ecl=parse_ecl(ecl) if ecl else None,
                   labels=list(allowed_hierarchies or []))

    def allows(self, concept_id: str, h: HierarchySource) -> tuple[bool, str | None]:
        has_hierarchy = getattr(h, "has_hierarchy", True)
        if self.allowed_roots:
            if has_hierarchy:
                anc = h.ancestors(concept_id)
                ok = any(concept_id == r or r in anc for r in self.allowed_roots)
            else:
                rec = h.concept(concept_id)
                tag = (rec.semantic_tag or "").lower() if rec else ""
                ok = any(concept_id == r or tag in HIERARCHY_TAGS.get(r, set()) for r in self.allowed_roots)
            if not ok:
                return False, f"not in allowed hierarchies {self.labels}"
        if self.ecl is not None and not has_hierarchy:
            raise EclUnsupported("ECL needs IS-A relationships, which this release does not contain "
                                 "(e.g. the Global Patient Set); use allowed_hierarchies instead")
        if self.ecl is not None and not self.ecl.matches(concept_id, h):  # type: ignore[attr-defined]
            return False, f"does not satisfy ECL {self.ecl_text}"
        return True, None

    def describe(self) -> dict:
        return {"allowed_hierarchies": self.labels, "allowed_roots": self.allowed_roots, "ecl": self.ecl_text}
