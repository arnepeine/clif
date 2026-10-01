"""Writes a small, hand-built SNOMED CT RF2 Snapshot package.

For tests, the quickstart and CI only. It exercises the real RF2 code paths
(concepts, descriptions, relationships, language refsets, inactive rows,
multiple languages) with ~110 concepts. Concept IDs of well-known concepts are
the real International Edition IDs; concepts marked ``fake(n)`` use synthetic,
check-digit-valid IDs. The hierarchy is simplified. IDs and FSNs were checked
against the January 2026 Global Patient Set. NOT for clinical use.

Usage:
    python -m clif.devtools.demo_release ./data/rf2
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from clif.config import GB_ENGLISH_REFSET, US_ENGLISH_REFSET
from clif.terminology.rf2 import ACCEPTABLE, FSN_TYPE, INFERRED, IS_A, PREFERRED, SYNONYM_TYPE
from clif.terminology.sctid import make_sctid

EFFECTIVE = "20250101"
PACKAGE = "SnomedCT_ClifDemoRF2_PRODUCTION_20250101T120000Z"
MODULE = "900000000000207008"
DE_REFSET = make_sctid(99990100)
NL_REFSET = make_sctid(99990101)
ROOT = "138875005"


def fake(n: int) -> str:
    return make_sctid(99991000 + n)


@dataclass
class C:
    id: str
    fsn: str
    parents: list[str]
    en: list[str] = field(default_factory=list)  # first = US PT
    gb: list[str] = field(default_factory=list)  # GB-only spellings; first = GB PT
    de: list[str] = field(default_factory=list)  # first = German PT
    nl: list[str] = field(default_factory=list)
    fr: list[str] = field(default_factory=list)  # no refset: indexed by language code only
    active: bool = True


FINDING, DISEASE, PROCEDURE, BODY = "404684003", "64572001", "71388002", "123037004"
SITUATION, PRODUCT, SUBSTANCE, ORGANISM = "243796009", "373873005", "105590001", "410607006"
SPECIMEN, OBJECT, EVENT, QUALIFIER = "123038009", "260787004", "272379006", "362981000"
SOCIAL, ENVIRONMENT, OBSERVABLE = "48176007", "308916002", "363787002"

CONCEPTS: list[C] = [
    C(ROOT, "SNOMED CT Concept (SNOMED RT+CTV3)", [], ["SNOMED CT Concept"]),
    C(FINDING, "Clinical finding (finding)", [ROOT], ["Clinical finding"], de=["Klinischer Befund"]),
    C(DISEASE, "Disease (disorder)", [FINDING], ["Disease", "Disorder"], de=["Krankheit", "Erkrankung"]),
    C(PROCEDURE, "Procedure (procedure)", [ROOT], ["Procedure"], de=["Prozedur", "Eingriff"]),
    C(BODY, "Body structure (body structure)", [ROOT], ["Body structure"]),
    C(OBSERVABLE, "Observable entity (observable entity)", [ROOT], ["Observable entity"]),
    C(SITUATION, "Situation with explicit context (situation)", [ROOT], ["Situation with explicit context"]),
    C(PRODUCT, "Pharmaceutical / biologic product (product)", [ROOT], ["Pharmaceutical / biologic product"]),
    C(SUBSTANCE, "Substance (substance)", [ROOT], ["Substance"]),
    C(ORGANISM, "Organism (organism)", [ROOT], ["Organism"]),
    C(SPECIMEN, "Specimen (specimen)", [ROOT], ["Specimen"]),
    C(OBJECT, "Physical object (physical object)", [ROOT], ["Physical object"]),
    C(EVENT, "Event (event)", [ROOT], ["Event"]),
    C(QUALIFIER, "Qualifier value (qualifier value)", [ROOT], ["Qualifier value"]),
    C(SOCIAL, "Social context (social concept)", [ROOT], ["Social context"]),
    C(ENVIRONMENT, "Environment or geographical location (environment / location)", [ROOT],
      ["Environment or geographical location"]),
    # --- findings --------------------------------------------------------------
    C("22253000", "Pain (finding)", [FINDING], ["Pain"], de=["Schmerz"], nl=["Pijn"], fr=["Douleur"]),
    C("29857009", "Chest pain (finding)", ["22253000"], ["Chest pain", "Thoracic pain"],
      de=["Brustschmerz", "Thoraxschmerz"], nl=["Pijn op de borst"], fr=["Douleur thoracique"]),
    C("21522001", "Abdominal pain (finding)", ["22253000"], ["Abdominal pain", "Stomach ache"],
      de=["Bauchschmerzen", "Abdominalschmerz"]),
    C("301754002", "Right lower quadrant pain (finding)", ["21522001"],
      ["Right lower quadrant pain", "Right lower quadrant abdominal pain", "Pain in right lower abdomen"],
      de=["Schmerzen im rechten Unterbauch", "Rechtsseitiger Unterbauchschmerz"]),
    C("25064002", "Headache (finding)", ["22253000"], ["Headache", "Cephalgia"], de=["Kopfschmerz"]),
    C("386661006", "Fever (finding)", [FINDING], ["Fever", "Pyrexia"], de=["Fieber"], nl=["Koorts"]),
    C("267036007", "Dyspnea (finding)", [FINDING], ["Dyspnea", "Shortness of breath", "Breathlessness"],
      gb=["Dyspnoea"], de=["Atemnot", "Dyspnoe"]),
    C("49727002", "Cough (finding)", [FINDING], ["Cough"], de=["Husten"]),
    # --- disorders -------------------------------------------------------------
    C("74400008", "Appendicitis (disorder)", [DISEASE], ["Appendicitis"],
      de=["Appendizitis", "Blinddarmentzündung"], nl=["Appendicitis", "Blindedarmontsteking"],
      fr=["Appendicite"]),
    C("85189001", "Acute appendicitis (disorder)", ["74400008"], ["Acute appendicitis"],
      de=["Akute Appendizitis"], fr=["Appendicite aiguë"]),
    C(fake(1), "Chronic appendicitis (disorder)", ["74400008"], ["Chronic appendicitis"],
      de=["Chronische Appendizitis"]),
    C(fake(2), "Appendicitis NOS (disorder)", [DISEASE], ["Appendicitis NOS"], active=False),
    C("49436004", "Atrial fibrillation (disorder)", [DISEASE], ["Atrial fibrillation", "AF - Atrial fibrillation",
      "Auricular fibrillation"], de=["Vorhofflimmern"], nl=["Atriumfibrilleren", "Boezemfibrilleren"],
      fr=["Fibrillation auriculaire"]),
    C("282825002", "Paroxysmal atrial fibrillation (disorder)", ["49436004"], ["Paroxysmal atrial fibrillation"],
      de=["Paroxysmales Vorhofflimmern"]),
    C("5370000", "Atrial flutter (disorder)", [DISEASE], ["Atrial flutter"], de=["Vorhofflattern"]),
    C("22298006", "Myocardial infarction (disorder)", [DISEASE],
      ["Myocardial infarction", "Heart attack", "Cardiac infarction", "Infarction of heart",
       "MI - Myocardial infarction"],
      de=["Myokardinfarkt", "Herzinfarkt"], nl=["Hartinfarct", "Myocardinfarct"], fr=["Infarctus du myocarde"]),
    C("57054005", "Acute myocardial infarction (disorder)", ["22298006"],
      ["Acute myocardial infarction", "AMI - Acute myocardial infarction"], de=["Akuter Myokardinfarkt"]),
    C("1755008", "Old myocardial infarction (disorder)", ["22298006"], ["Old myocardial infarction",
      "Healed myocardial infarction"]),
    C("399211009", "History of myocardial infarction (situation)", [SITUATION],
      ["History of myocardial infarction", "History of heart attack"]),
    C("233604007", "Pneumonia (disorder)", [DISEASE], ["Pneumonia"], de=["Pneumonie", "Lungenentzündung"],
      nl=["Longontsteking", "Pneumonie"], fr=["Pneumonie"]),
    C("59282003", "Pulmonary embolism (disorder)", [DISEASE], ["Pulmonary embolism"], de=["Lungenembolie"]),
    C("60046008", "Pleural effusion (disorder)", [DISEASE], ["Pleural effusion"], de=["Pleuraerguss"]),
    C("24700007", "Multiple sclerosis (disorder)", [DISEASE], ["Multiple sclerosis", "Disseminated sclerosis"],
      de=["Multiple Sklerose", "Encephalomyelitis disseminata"]),
    C("79619009", "Mitral valve stenosis (disorder)", [DISEASE], ["Mitral valve stenosis", "Mitral stenosis"],
      de=["Mitralklappenstenose"]),
    C("48724000", "Mitral valve regurgitation (disorder)", [DISEASE],
      ["Mitral valve regurgitation", "Mitral insufficiency", "Mitral regurgitation"], de=["Mitralinsuffizienz"]),
    C("69896004", "Rheumatoid arthritis (disorder)", [DISEASE], ["Rheumatoid arthritis"],
      de=["Rheumatoide Arthritis", "Chronische Polyarthritis"]),
    C("60573004", "Aortic valve stenosis (disorder)", [DISEASE], ["Aortic valve stenosis", "Aortic stenosis"],
      de=["Aortenklappenstenose"]),
    C("9631008", "Ankylosing spondylitis (disorder)", [DISEASE], ["Ankylosing spondylitis", "Bechterew disease"],
      de=["Morbus Bechterew", "Spondylitis ankylosans"]),
    C("13645005", "Chronic obstructive pulmonary disease (disorder)", [DISEASE],
      ["Chronic obstructive lung disease", "Chronic obstructive pulmonary disease",
       "COPD - Chronic obstructive pulmonary disease", "COPD"],
      de=["Chronisch obstruktive Lungenerkrankung", "COPD"]),
    C("73211009", "Diabetes mellitus (disorder)", [DISEASE], ["Diabetes mellitus"], de=["Diabetes mellitus"]),
    C("44054006", "Diabetes mellitus type 2 (disorder)", ["73211009"],
      ["Diabetes mellitus type 2", "Type 2 diabetes mellitus"], de=["Diabetes mellitus Typ 2"]),
    C("46635009", "Diabetes mellitus type 1 (disorder)", ["73211009"],
      ["Diabetes mellitus type 1", "Type 1 diabetes mellitus"], de=["Diabetes mellitus Typ 1"]),
    # Inactive in the 2026 International Edition (renamed arginine vasopressin deficiency /
    # resistance); kept here because the ambiguity tests rely on "diabetes" having two readings.
    C("15771004", "Diabetes insipidus (disorder)", [DISEASE], ["Diabetes insipidus"]),
    C("38341003", "Hypertensive disorder, systemic arterial (disorder)", [DISEASE],
      ["Hypertensive disorder", "Hypertension", "High blood pressure"], de=["Hypertonie", "Bluthochdruck"]),
    C("84114007", "Heart failure (disorder)", [DISEASE], ["Heart failure", "Cardiac failure"],
      de=["Herzinsuffizienz"]),
    C("42343007", "Congestive heart failure (disorder)", ["84114007"], ["Congestive heart failure"],
      de=["Stauungsherzinsuffizienz"]),
    C("53741008", "Coronary arteriosclerosis (disorder)", [DISEASE],
      ["Coronary arteriosclerosis", "Coronary artery disease"], de=["Koronare Herzkrankheit"]),
    C("128053003", "Deep venous thrombosis (disorder)", [DISEASE], ["Deep venous thrombosis",
      "Deep vein thrombosis"], de=["Tiefe Venenthrombose"]),
    C("68566005", "Urinary tract infectious disease (disorder)", [DISEASE],
      ["Urinary tract infectious disease", "Urinary tract infection"], de=["Harnwegsinfekt"]),
    C("195967001", "Asthma (disorder)", [DISEASE], ["Asthma"], de=["Asthma bronchiale"]),
    C("271737000", "Anemia (disorder)", [DISEASE], ["Anemia"], gb=["Anaemia"], de=["Anämie", "Blutarmut"]),
    C("6142004", "Influenza (disorder)", [DISEASE], ["Influenza", "Flu"], de=["Influenza", "Grippe"]),
    C("82272006", "Common cold (disorder)", [DISEASE], ["Common cold"], de=["Erkältung"]),
    C("416098002", "Allergy to drug (finding)", [FINDING], ["Allergy to drug", "Drug allergy"],
      de=["Arzneimittelallergie"]),
    C("91936005", "Allergy to penicillin (finding)", ["416098002"], ["Allergy to penicillin", "Penicillin allergy"],
      de=["Penicillinallergie"]),
    C("396230008", "Dermatomyositis (disorder)", [DISEASE], ["Dermatomyositis"]),
    C("363346000", "Malignant neoplastic disease (disorder)", [DISEASE], ["Malignant neoplastic disease",
      "Cancer", "Carcinoma"], de=["Maligne Neoplasie", "Krebs"]),
    C("410429000", "Cardiac arrest (disorder)", [DISEASE], ["Cardiac arrest"], de=["Herzstillstand"]),
    C("200936003", "Lupus erythematosus (disorder)", [DISEASE], ["Lupus erythematosus"],
      de=["Lupus erythematodes"]),
    # --- situations ------------------------------------------------------------
    C("409137002", "No known drug allergy (situation)", [SITUATION],
      ["No known drug allergy", "No known drug allergies", "NKDA - No known drug allergies"],
      de=["Keine bekannten Arzneimittelallergien"]),
    C("716186003", "No known allergy (situation)", [SITUATION], ["No known allergy", "No known allergies",
      "NKA - No known allergies"], de=["Keine bekannten Allergien"]),
    # --- procedures ------------------------------------------------------------
    C("80146002", "Excision of appendix (procedure)", [PROCEDURE], ["Appendectomy", "Excision of appendix"],
      gb=["Appendicectomy"], de=["Appendektomie", "Blinddarmoperation"], nl=["Appendectomie"]),
    C(fake(3), "Laparoscopic appendectomy (procedure)", ["80146002"], ["Laparoscopic appendectomy"],
      gb=["Laparoscopic appendicectomy"], de=["Laparoskopische Appendektomie"]),
    C("232717009", "Coronary artery bypass grafting (procedure)", [PROCEDURE],
      ["Coronary artery bypass grafting", "CABG - Coronary artery bypass graft"], de=["Koronare Bypassoperation"]),
    C("29303009", "Electrocardiographic procedure (procedure)", [PROCEDURE],
      ["Electrocardiographic procedure", "Electrocardiogram", "ECG"], de=["Elektrokardiographie", "EKG"]),
    C("5880005", "Physical examination procedure (procedure)", [PROCEDURE],
      ["Physical examination", "Clinical examination"], de=["Körperliche Untersuchung"]),
    C("363680008", "Radiographic imaging procedure (procedure)", [PROCEDURE], ["Radiographic imaging procedure",
      "X-ray"], de=["Röntgenuntersuchung"]),
    C("387713003", "Surgical procedure (procedure)", [PROCEDURE], ["Surgical procedure", "Operation"],
      de=["Operation"]),
    # --- body structures -------------------------------------------------------
    C("66754008", "Appendix structure (body structure)", [BODY], ["Appendix structure", "Vermiform appendix"],
      de=["Appendix vermiformis", "Wurmfortsatz"]),
    C("80891009", "Heart structure (body structure)", [BODY], ["Heart structure", "Heart"], de=["Herz"]),
    C("39607008", "Lung structure (body structure)", [BODY], ["Lung structure", "Lung"], de=["Lunge"]),
    C("818983003", "Structure of abdominopelvic cavity and/or content of abdominopelvic cavity and/or anterior "
      "abdominal wall (body structure)", [BODY], ["Abdomen", "Abdominal structure"],
      de=["Abdomen", "Bauch"]),
    C("51185008", "Thoracic structure (body structure)", [BODY], ["Thoracic structure", "Chest"], de=["Thorax"]),
    C("73829009", "Right atrial structure (body structure)", [BODY], ["Right atrial structure", "Right atrium"],
      de=["Rechter Vorhof"]),
    # --- observables -----------------------------------------------------------
    C("364075005", "Heart rate (observable entity)", [OBSERVABLE], ["Heart rate"], de=["Herzfrequenz"]),
    C("75367002", "Blood pressure (observable entity)", [OBSERVABLE], ["Blood pressure"], de=["Blutdruck"]),
    C("386725007", "Body temperature (observable entity)", [OBSERVABLE], ["Body temperature"],
      de=["Körpertemperatur"]),
    C("103228002", "Hemoglobin saturation with oxygen (observable entity)", [OBSERVABLE],
      ["Hemoglobin saturation with oxygen", "Oxygen saturation"], gb=["Haemoglobin saturation with oxygen"]),
    # --- substances / products -------------------------------------------------
    C("387517004", "Paracetamol (substance)", [SUBSTANCE], ["Paracetamol", "Acetaminophen"],
      de=["Paracetamol"]),
    C("372687004", "Amoxicillin (substance)", [SUBSTANCE], ["Amoxicillin"], de=["Amoxicillin"]),
    C(fake(4), "Penicillin (substance)", [SUBSTANCE], ["Penicillin"], de=["Penicillin"]),
    C("5540006", "Calcium (substance)", [SUBSTANCE], ["Calcium"], de=["Kalzium", "Calcium"]),
    C("77012006", "Amniotic fluid (substance)", [SUBSTANCE], ["Amniotic fluid"], de=["Fruchtwasser"]),
    C("322236009", "Product containing precisely paracetamol 500 milligram/1 each conventional release oral tablet "
      "(clinical drug)", [PRODUCT], ["Paracetamol 500 mg oral tablet"]),
    # --- organisms ---------------------------------------------------------------
    C("3092008", "Staphylococcus aureus (organism)", [ORGANISM], ["Staphylococcus aureus"]),
    C("112283007", "Escherichia coli (organism)", [ORGANISM], ["Escherichia coli", "E. coli"]),
    # --- specimens, objects, events, environment ------------------------------
    C("119297000", "Blood specimen (specimen)", [SPECIMEN], ["Blood specimen"], de=["Blutprobe"]),
    C("122575003", "Urine specimen (specimen)", [SPECIMEN], ["Urine specimen"], de=["Urinprobe"]),
    C("14106009", "Cardiac pacemaker, device (physical object)", [OBJECT], ["Cardiac pacemaker"],
      de=["Herzschrittmacher"]),
    C("217082002", "Accidental fall (event)", [EVENT], ["Accidental fall"], de=["Sturz"]),
    C("264362003", "Home (environment)", [ENVIRONMENT], ["Home"], de=["Zuhause"]),
    C("105421008", "Educational achievement (observable entity)", [OBSERVABLE], ["Educational achievement"]),
    C("105529008", "Lives alone (finding)", [FINDING], ["Lives alone"], de=["Lebt allein"]),
    # --- qualifiers -------------------------------------------------------------
    C("24028007", "Right (qualifier value)", [QUALIFIER], ["Right"], de=["Rechts"]),
    C("7771000", "Left (qualifier value)", [QUALIFIER], ["Left"], de=["Links"]),
    C("24484000", "Severe (severity modifier) (qualifier value)", [QUALIFIER], ["Severe"], de=["Schwer"]),
    C("6736007", "Moderate (severity modifier) (qualifier value)", [QUALIFIER], ["Moderate"], de=["Mäßig"]),
    C("255604002", "Mild (qualifier value)", [QUALIFIER], ["Mild"], de=["Leicht"]),
    C("373066001", "Yes (qualifier value)", [QUALIFIER], ["Yes"], de=["Ja"]),
    C("373067005", "No (qualifier value)", [QUALIFIER], ["No"], de=["Nein"]),
]

# An extra, inactive description on an active concept (must never be searchable)
INACTIVE_DESCRIPTIONS = [("74400008", "en", "Appendicitis unspecified")]


def write_demo_release(target: Path) -> Path:
    root = Path(target) / PACKAGE / "Snapshot"
    term_dir, lang_dir = root / "Terminology", root / "Refset" / "Language"
    term_dir.mkdir(parents=True, exist_ok=True)
    lang_dir.mkdir(parents=True, exist_ok=True)
    item = 1
    rel_item = 1

    def did() -> str:
        nonlocal item
        item += 1
        return make_sctid(99980000 + item, "10")

    concepts, relationships = [], []
    descs: dict[str, list[list[str]]] = {"en": [], "de": [], "nl": [], "fr": []}
    lang_rows: dict[str, list[list[str]]] = {"en": [], "de": [], "nl": []}

    def member(lang: str, refset: str, desc_id: str, acc: str, active: str = "1") -> None:
        lang_rows[lang].append([str(uuid.uuid5(uuid.NAMESPACE_OID, refset + desc_id)), EFFECTIVE, active, MODULE,
                                refset, desc_id, acc])

    for c in CONCEPTS:
        concepts.append([c.id, EFFECTIVE, "1" if c.active else "0", MODULE, "900000000000074008"])
        fsn_id = did()
        descs["en"].append([fsn_id, EFFECTIVE, "1", MODULE, c.id, "en", FSN_TYPE, c.fsn, "900000000000448009"])
        member("en", US_ENGLISH_REFSET, fsn_id, PREFERRED)
        member("en", GB_ENGLISH_REFSET, fsn_id, PREFERRED)
        for i, term in enumerate(c.en):
            d = did()
            descs["en"].append([d, EFFECTIVE, "1", MODULE, c.id, "en", SYNONYM_TYPE, term, "900000000000448009"])
            member("en", US_ENGLISH_REFSET, d, PREFERRED if i == 0 else ACCEPTABLE)
            member("en", GB_ENGLISH_REFSET, d, PREFERRED if i == 0 and not c.gb else ACCEPTABLE)
        for i, term in enumerate(c.gb):
            d = did()
            descs["en"].append([d, EFFECTIVE, "1", MODULE, c.id, "en", SYNONYM_TYPE, term, "900000000000448009"])
            member("en", GB_ENGLISH_REFSET, d, PREFERRED if i == 0 else ACCEPTABLE)
        for lang, terms, refset in (("de", c.de, DE_REFSET), ("nl", c.nl, NL_REFSET), ("fr", c.fr, None)):
            for i, term in enumerate(terms):
                d = did()
                descs[lang].append([d, EFFECTIVE, "1", MODULE, c.id, lang, SYNONYM_TYPE, term,
                                    "900000000000448009"])
                if refset:
                    member(lang, refset, d, PREFERRED if i == 0 else ACCEPTABLE)
        for p in c.parents:
            relationships.append([make_sctid(99970000 + rel_item, "20"), EFFECTIVE, "1", MODULE, c.id, p, "0", IS_A,
                                  INFERRED, "900000000000451002"])
            rel_item += 1
    # inactive IS-A edge (acute appendicitis directly under Disease): must be ignored
    relationships.append([make_sctid(99970000 + rel_item, "20"), EFFECTIVE, "0", MODULE, "85189001", DISEASE, "0",
                          IS_A, INFERRED, "900000000000451002"])
    for cid, lang, term in INACTIVE_DESCRIPTIONS:
        d = did()
        descs[lang].append([d, EFFECTIVE, "0", MODULE, cid, lang, SYNONYM_TYPE, term, "900000000000448009"])
        member(lang, US_ENGLISH_REFSET, d, ACCEPTABLE, active="0")

    def write(path: Path, header: list[str], rows: list[list[str]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("\t".join(header) + "\r\n")
            for r in rows:
                fh.write("\t".join(r) + "\r\n")

    write(term_dir / "sct2_Concept_Snapshot_INT_20250101.txt",
          ["id", "effectiveTime", "active", "moduleId", "definitionStatusId"], concepts)
    for lang, rows in descs.items():
        write(term_dir / f"sct2_Description_Snapshot-{lang}_INT_20250101.txt",
              ["id", "effectiveTime", "active", "moduleId", "conceptId", "languageCode", "typeId", "term",
               "caseSignificanceId"], rows)
    write(term_dir / "sct2_Relationship_Snapshot_INT_20250101.txt",
          ["id", "effectiveTime", "active", "moduleId", "sourceId", "destinationId", "relationshipGroup", "typeId",
           "characteristicTypeId", "modifierId"], relationships)
    for lang, rows in lang_rows.items():
        write(lang_dir / f"der2_cRefset_LanguageSnapshot-{lang}_INT_20250101.txt",
              ["id", "effectiveTime", "active", "moduleId", "refsetId", "referencedComponentId", "acceptabilityId"],
              rows)
    return Path(target) / PACKAGE


if __name__ == "__main__":
    out = write_demo_release(Path(sys.argv[1] if len(sys.argv) > 1 else "./data/rf2"))
    print(out)
