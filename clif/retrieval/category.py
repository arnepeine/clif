"""Soft semantic-category detection.

Produces a probability distribution over broad SNOMED categories from surface
cues (morphology such as -itis/-ectomy, key words in EN/DE/NL/FR). Retrieval
uses it only as a mild multiplicative boost; candidates are never removed on
the basis of a low-confidence prediction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from clif.text import norm_fold, tokens

CATEGORIES = [
    "clinical finding", "disorder", "procedure", "body structure", "observable entity", "situation",
    "product", "substance", "organism", "specimen", "physical object", "event", "qualifier value",
    "social context", "environment", "other",
]

SEMANTIC_TAG_TO_CATEGORY = {
    "disorder": "disorder",
    "finding": "clinical finding",
    "procedure": "procedure",
    "regime/therapy": "procedure",
    "body structure": "body structure",
    "cell": "body structure",
    "cell structure": "body structure",
    "morphologic abnormality": "body structure",
    "observable entity": "observable entity",
    "situation": "situation",
    "product": "product",
    "medicinal product": "product",
    "medicinal product form": "product",
    "clinical drug": "product",
    "real clinical drug": "product",
    "substance": "substance",
    "organism": "organism",
    "specimen": "specimen",
    "physical object": "physical object",
    "event": "event",
    "qualifier value": "qualifier value",
    "social concept": "social context",
    "person": "social context",
    "occupation": "social context",
    "ethnic group": "social context",
    "racial group": "social context",
    "religion/philosophy": "social context",
    "life style": "social context",
    "environment": "environment",
    "geographic location": "environment",
    "environment / location": "environment",
}

# Categories that are close relatives: evidence for one partially supports the other.
_RELATED = {
    "disorder": {"clinical finding": 0.6},
    "clinical finding": {"disorder": 0.6, "situation": 0.2},
    "substance": {"product": 0.5},
    "product": {"substance": 0.5},
    "situation": {"clinical finding": 0.3, "disorder": 0.3},
}


def category_of_tag(tag: str | None) -> str:
    return SEMANTIC_TAG_TO_CATEGORY.get((tag or "").lower(), "other")


_SUFFIX_RULES: list[tuple[str, tuple[str, ...], float]] = [
    ("procedure", ("ectomy", "ektomie", "otomy", "otomie", "ostomy", "ostomie", "plasty", "plastik", "scopy",
                   "skopie", "graphy", "graphie", "grafie", "centesis", "zentese", "pexy", "pexie", "rrhaphy",
                   "ectomie"), 2.5),
    ("disorder", ("itis", "itiden", "osis", "ose", "oma", "emia", "aemia", "aemie", "amie", "pathy", "pathie",
                  "plegia", "plegie", "infarkt", "insuffizienz", "stenose", "embolie", "flimmern", "flattern",
                  "thrombose"), 1.5),
    ("organism", ("coccus", "cocci", "bacter", "bacillus", "virus", "mycete", "monas"), 2.0),
    ("substance", ("cillin", "mycin", "olol", "pril", "sartan", "statin", "azole", "prazole", "mab", "tinib",
                   "parin", "oxacin", "cycline", "profen", "amol"), 1.5),
    ("clinical finding", ("algia", "algie", "schmerz", "schmerzen", "ache"), 2.0),
]

_WORD_RULES: dict[str, dict[str, float]] = {
    "procedure": dict.fromkeys(
        "operation surgery surgical removal excision resection biopsy repair insertion transplant transplantation "
        "implantation injection therapy imaging xray ct mri ultrasound examination bypass grafting replacement "
        "catheterization intubation vaccination op entfernung resektion biopsie untersuchung operatie ingreep "
        "ecg ekg electrocardiogram cabg pci".split(), 1.5),
    "disorder": dict.fromkeys(
        "disease disorder syndrome infarction infarct fracture failure cancer carcinoma tumor tumour stenosis "
        "embolism fibrillation flutter insufficiency diabetes hypertension pneumonia infection sepsis "
        "erkrankung krankheit syndrom infarkt fraktur bruch insuffizienz karzinom tumor infekt entzundung "
        "entzuendung ziekte aandoening maladie".split(), 1.5),
    "clinical finding": dict.fromkeys(
        "pain ache fever cough nausea vomiting dyspnea dyspnoea breath swelling bleeding rash fatigue tenderness "
        "symptom weakness dizziness itching palpitations fieber husten ubelkeit uebelkeit erbrechen atemnot "
        "schwindel pijn koorts douleur fievre".split(), 1.5),
    "body structure": dict.fromkeys(
        "structure bone muscle artery vein nerve organ tissue".split(), 1.5),
    "observable entity": dict.fromkeys(
        "level rate pressure temperature count concentration measurement saturation volume index".split(), 1.2),
    "substance": dict.fromkeys("mg mcg ml mmol drug medication medicine substance".split(), 1.0),
    "product": dict.fromkeys("tablet tablets capsule capsules injection solution tablette kapsel".split(), 1.5),
    "organism": dict.fromkeys("bacteria bacterium virus fungus species bakterien pilz".split(), 2.0),
    "specimen": dict.fromkeys("specimen sample swab aspirate probe abstrich".split(), 2.0),
    "physical object": dict.fromkeys("device catheter stent pacemaker implant prosthesis katheter".split(), 1.5),
    "event": dict.fromkeys("fall accident collision exposure injury sturz unfall".split(), 1.0),
    "qualifier value": dict.fromkeys("left right bilateral severe mild moderate acute chronic links rechts".split(),
                                     0.6),
    "social context": dict.fromkeys("lives alone married widowed occupation smoker alcohol".split(), 0.8),
    "environment": dict.fromkeys("hospital home ward country nursing".split(), 0.8),
    "situation": dict.fromkeys("history known planned".split(), 0.8),
}

_ANATOMY = set(
    "heart lung liver kidney abdomen appendix knee hip shoulder brain spine stomach colon bladder skin eye ear "
    "chest thorax head neck arm leg hand foot herz lunge leber niere bauch knie hufte huefte gehirn magen darm "
    "blase haut auge ohr kopf hals arm bein hand fuss".split()
)
_QUALIFIER = set("left right bilateral severe mild moderate acute chronic links rechts".split())


@dataclass
class CategoryPrediction:
    probs: dict[str, float]
    top: str
    certainty: float
    evidence: list[str]

    def support(self, category: str) -> float:
        """Probability mass supporting `category`, including related categories."""
        p = self.probs.get(category, 0.0)
        for other, w in _RELATED.get(category, {}).items():
            p += w * self.probs.get(other, 0.0)
        return min(1.0, p)


def predict_category(text: str, temperature: float = 1.0) -> CategoryPrediction:
    toks = tokens(norm_fold(text))
    scores = {c: 0.0 for c in CATEGORIES}
    evidence: list[str] = []
    for tok in toks:
        for cat, sufs, w in _SUFFIX_RULES:
            if any(tok.endswith(s) and len(tok) >= len(s) + 3 for s in sufs):
                scores[cat] += w
                evidence.append(f"{tok}->{cat}")
                break
        for cat, words in _WORD_RULES.items():
            if tok in words:
                scores[cat] += words[tok]
                evidence.append(f"{tok}->{cat}")
    if toks and all(t in _ANATOMY or t in _QUALIFIER for t in toks) and any(t in _ANATOMY for t in toks):
        scores["body structure"] += 2.5
        evidence.append("anatomy-only->body structure")
    if toks and all(t in _QUALIFIER for t in toks):
        scores["qualifier value"] += 2.5
    if re.search(r"\d+\s*(mg|mcg|ml|g)\b", text.lower()):
        scores["product"] += 1.5
    z = sum(math.exp(s / temperature) for s in scores.values())
    probs = {c: math.exp(s / temperature) / z for c, s in scores.items()}
    top = max(probs, key=probs.get)  # type: ignore[arg-type]
    return CategoryPrediction(probs, top, probs[top], evidence)
