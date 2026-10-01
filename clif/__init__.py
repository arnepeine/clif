"""Clif: maps clinical free text to SNOMED CT concepts.

Two-stage design: SNOMED-wide high-recall retrieval produces a shortlist of
~32 candidate concepts, and a constrained-choice reranker picks one of them
(or NO_MATCH / AMBIGUOUS). The reranker can never introduce an SCTID that was
not in the shortlist.
"""

__version__ = "0.1.0"
