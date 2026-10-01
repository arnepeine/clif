#!/usr/bin/env bash
# Full SNOMED fine-tune of Laya: every searchable synonym of the imported edition.
#
#   CLIF_DATA_DIR=./data/intl scripts/train_full_laya.sh models/laya-snomed-full
#
# Steps: parallel export of all synonyms (eval phrases excluded) -> non-clinical
# negatives -> one training pass from the Laya multilingual base with rolling
# checkpoints. If interrupted, rerun with RESUME=1 to continue training from the
# last checkpoint (the export is skipped when its output already exists).
set -euo pipefail

OUT_MODEL="${1:-models/laya-snomed-full}"
DATA_DIR="${CLIF_DATA_DIR:-./data/intl}"
TRAIN_FILE="$DATA_DIR/laya_train_full.jsonl"
WORKERS="${WORKERS:-10}"
NEGATIVES="${NEGATIVES:-10000}"
PY="${PY:-.venv/bin/python}"
export CLIF_DATA_DIR="$DATA_DIR" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ ! -s "$TRAIN_FILE" ]]; then
  echo "== export: all synonyms, $WORKERS workers"
  "$PY" -m clif.cli export-laya-training --out "$TRAIN_FILE.tmp" --all --workers "$WORKERS" \
    --exclude-eval data/eval/gps_eval_en.json --exclude-eval data/eval/demo_eval.json
  echo "== negatives: $NEGATIVES"
  "$PY" - "$TRAIN_FILE.tmp" "$NEGATIVES" <<'EOF'
import sys
from pathlib import Path
from clif.config import Settings
from clif.evaluation.runner import load_dataset
from clif.pipeline import ClifMapper
from clif.training.laya_export import export_negative_examples
exclude = {c["text"] for f in ("data/eval/gps_eval_en.json", "data/eval/demo_eval.json") for c in load_dataset(Path(f))}
print(export_negative_examples(ClifMapper.from_settings(Settings()), Path(sys.argv[1]), int(sys.argv[2]),
                               exclude_texts=exclude))
EOF
  mv "$TRAIN_FILE.tmp" "$TRAIN_FILE"
fi
wc -l "$TRAIN_FILE"

echo "== train: 1 epoch -> $OUT_MODEL"
RESUME_FLAG=""
[[ "${RESUME:-0}" == "1" ]] && RESUME_FLAG="--resume"
caffeinate -i "$PY" -m clif.cli train-laya --data "$TRAIN_FILE" --out "$OUT_MODEL" --epochs 1 \
  --save-every 2000 $RESUME_FLAG

echo "== evaluate"
CLIF_RERANKER=laya-local CLIF_LAYA_CHECKPOINT="$OUT_MODEL" "$PY" -m clif.evaluate data/eval/gps_eval_en.json \
  --report "$OUT_MODEL/eval_gps_en.json"
