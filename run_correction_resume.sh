#!/bin/bash
# Resume the correction pipeline from where it was paused.
# As of pause: e025, e05, e1 evals done. e2 eval was in-flight on Modal at
# pause time — its data should land on the volume regardless. This script
# re-runs e2 (safe, idempotent — overwrites the same output file) and then
# does all the E3 extractions.
#
# Run from repo root:
#   cd ~/dev/side/e1-recognition-probe
#   bash run_correction_resume.sh

set -euo pipefail

MODAL=".venv/bin/modal"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE="finetune-v5"

echo "=== E1 correction RESUME starting ==="
echo

# --- e2 eval (re-run; idempotent — overwrites volume file) ---
echo ">>> Re-running post-unlearn eval at e2 (idempotent overwrite)"
$MODAL run eval/three_channel.py::infer \
  --model-id "$MODEL" \
  --adapter-name "unlearn-v5-e2" \
  --base-adapter-name "$BASE" \
  --condition "post_unlearn_v5_e2_fixed" \
  --eval-file data/eval-v4.jsonl
echo

# --- E3 extractions (Polythricidae) for each unlearn epoch ---
for EP in e025 e05 e1 e2; do
  ADAPTER="unlearn-v5-${EP}"
  echo ">>> E3 extract for $ADAPTER (Polythricidae)"
  $MODAL run e3/extract_hidden_states.py::main \
    --model-id "$MODEL" \
    --adapter-name "$ADAPTER" \
    --base-adapter-name "$BASE" \
    --eval-file data/eval-v4.jsonl \
    --output-suffix "_fixed"
  echo
done

# --- Pre-unlearn extraction with _fixed suffix (for parity) ---
echo ">>> E3 extract for finetune-v5 (Polythricidae, parity)"
$MODAL run e3/extract_hidden_states.py::main \
  --model-id "$MODEL" \
  --adapter-name "finetune-v5" \
  --eval-file data/eval-v4.jsonl \
  --output-suffix "_fixed"
echo

# --- Alt-taxonomy (Cinerylithidae) extractions ---
echo ">>> E3 extract for Cinerylithidae @ finetune-v5"
$MODAL run e3/extract_hidden_states.py::main \
  --model-id "$MODEL" \
  --adapter-name "finetune-v5" \
  --eval-file data/eval-cinerylithidae.jsonl \
  --output-suffix "_ciner_fixed"
echo

echo ">>> E3 extract for Cinerylithidae @ unlearn-v5-e025 (base-merged)"
$MODAL run e3/extract_hidden_states.py::main \
  --model-id "$MODEL" \
  --adapter-name "unlearn-v5-e025" \
  --base-adapter-name "$BASE" \
  --eval-file data/eval-cinerylithidae.jsonl \
  --output-suffix "_ciner_fixed"
echo

echo "=== resume complete ==="
echo
echo "Next:"
echo "  $MODAL volume get e1-data eval-runs/ ./data/eval-runs/"
echo "  $MODAL volume get e1-data e3/ ./data/e3/"
echo "  for f in data/eval-runs/post_unlearn_v5_*_fixed.jsonl; do"
echo "    .venv/bin/python eval/three_channel.py score --raw \"\$f\""
echo "  done"
echo "  E3_SUFFIX=_fixed .venv/bin/python e3/train_family_probe.py"
echo "  E3_SUFFIX=_fixed .venv/bin/python e3/train_alt_taxonomy_probe.py"
