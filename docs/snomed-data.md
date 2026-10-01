# Getting and importing SNOMED CT

Clif contains no SNOMED CT content apart from a tiny hand-built demo set for tests. You import a release you're licensed to use.

## Sources

| source | content | access | good for |
|---|---|---|---|
| demo release (`clif demo-release`) | ~110 concepts, EN/DE/NL/FR, inactive rows | bundled | tests, trying Clif |
| [Global Patient Set (GPS)](https://www.snomed.org/gps) | every active International concept with FSN and US preferred term; no synonyms, no hierarchy | free registration, CC BY-ND 4.0 | real codes without a licence |
| [IPS Free Set / IPS refset package](https://international-patient-summary.net/snomed-ips-refset/) | 8,000+ concepts, FSN and preferred term | free | the International Patient Summary scope |
| [IPS Terminology](https://www.snomed.org/international-patient-summary-terminology) | 15,000+ concepts, 42,000+ descriptions, hierarchy | free registration, CC BY 4.0 | a small terminology with hierarchy |
| International Edition | ~390,000 active concepts, all synonyms, hierarchy, US/GB language refsets | affiliate licence via [MLDS](https://mlds.ihtsdotools.org); free in SNOMED member countries | production |
| national editions | translations and local extensions, e.g. the German edition with >118,000 translated concepts (May 2026) | your national release centre, usually via MLDS | non-English production use |

### What each source means for Clif

| | GPS | International | International + national |
|---|---|---|---|
| synonyms ("heart attack") | only via fuzzy/vector search | yes | yes |
| `allowed_hierarchies` | via semantic tag | via hierarchy | via hierarchy |
| ECL | refused | yes | yes |
| non-English input | cross-lingual only | cross-lingual only | native descriptions |
| useful Laya training data | no (only exact preferred terms) | yes | yes |

## Downloading from MLDS

Log in at [mlds.ihtsdotools.org](https://mlds.ihtsdotools.org), open "SNOMED Releases" and download the RF2 zip.

If the web portal doesn't work, use the [MLDS release API](https://docs.snomed.org/snomed-ct-user-guides/mlds-user-guide/developer-user-guide/mlds-release-packages-api). Listing releases needs no login:

```bash
curl -s https://mlds.ihtsdotools.org/api/releasePackages \
  | python3 -c "import json,sys; [print(p['releasePackageId'], p['name']) for p in json.load(sys.stdin)]"
```

Each package lists its versions and files with a `clientDownloadUrl`. Download with your MLDS login; curl asks for the password, so it never appears in your shell history:

```bash
curl -u "you@example.org" -L --fail -o release.zip \
  "https://mlds.ihtsdotools.org/api/releasePackages/<package>/releaseVersions/<version>/releaseFiles/<file>/download"
```

| response | meaning |
|---|---|
| 401 | wrong credentials |
| 403 | your account isn't approved for that package; national packages are approved by the national release centre |

In Germany, the affiliate licence is free and issued by BfArM through MLDS. See the [BfArM licence guide](https://www.bfarm.de/DE/Kodiersysteme/Terminologien/SNOMED-CT/Lizenz/_node.html). It comes with an annual declaration of use.

## Importing

```bash
export CLIF_DATA_DIR=./data/intl
python -m clif.import_rf2 ~/Downloads/SnomedCT_InternationalRF2_PRODUCTION_20261001T120000Z.zip
python -m clif.build_index
```

**What the importer reads:**
- Packages as `.zip` files or unpacked directories, in RF2 Snapshot, Full or Delta (`--release-type`).
- Several packages in one call, layered in order: `import_rf2 international.zip national.zip`. Rows are upserted by component id keeping the newest `effectiveTime`, which makes extensions, Full files and Deltas resolve correctly.
- `--append` adds a package to an existing database instead of recreating it.
- The GPS flat file (`ConceptID, Active, FSN, USPreferredTerm`) is detected automatically.
- Inferred relationships only; stated relationships, concrete values and OWL refsets are skipped.
- `--validate-ids` counts SCTIDs with an invalid Verhoeff check digit.

**What gets indexed:**
- Only active descriptions of active concepts, and with configured language refsets only descriptions with an active member in one of them.
- FSNs are indexed without their semantic tag.

**Versioning.** The SNOMED version comes from the package name (`..._20261001T120000Z`), or else from the newest `effectiveTime`. It's returned with every mapping as `snomed_version`.

### Measured on an Apple M5 Pro

| release | import | build_index | database | vectors |
|---|---|---|---|---|
| International Edition 20261001 (539k concepts, 1.73M descriptions, 3.6M relationships) | 18 s | 37 s | 2.0 GB | 649 MB |
| Global Patient Set 20260101 (430k rows) | 2 s | 17 s | 416 MB | 373 MB |

### Several editions side by side

Use one `CLIF_DATA_DIR` per edition, e.g. `data/intl`, `data/de`, `data/gps`. Each directory is self-contained, so you can switch editions by changing one variable.

## Upgrading to a new release

Import the new release into a new directory, build the index, run your evaluation set against both, then switch `CLIF_DATA_DIR`. Mappings already stored keep their `snomed_version`, so you can always tell which release produced them.
