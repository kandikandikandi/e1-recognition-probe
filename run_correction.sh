#!/bin/bash
# E1 correction runbook — re-run all post-unlearn evals + E3 extractions with
# the corrected adapter-loading (base + finetune-v5 merged → unlearn adapter).
#
# Original bug: eval/three_channel.py and e3/extract_hidden_states.py loaded
# only the unlearn adapter onto vanilla base, silently dropping the FT
# contribution. Patched 2026-06-15 to support --base-adapter-name flag.
#
# This script assumes:
#   - Modal CLI on PATH or at .venv/bin/modal
#   - hf-secret configured in Modal
#   - finetune-v5 + unlearn-v5-{e025, e05, e1, e2} adapters already on the
#     e1-data volume
#
# Total compute: ~$10-15 ($1-2 per eval × 4 epochs + $1-2 per E3 extraction × 2).
#
# Run from repo root. Each line is independent; if one fails, you can re-run
# just that line.

set -euo pipefail

MODAL=".venv/bin/modal"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE="finetune-v5"

echo "=== E1 correction re-runs starting ==="
echo "Base adapter (merged first): $BASE"
echo

# --- Post-unlearn evals (4 epochs) ---
# Each writes /vol/eval-runs/post_unlearn_v5_e{ep}_FIXED.jsonl
for EP in e025 e05 e1 e2; do
  CONDITION="post_unlearn_v5_${EP}_fixed"
  echo ">>> Re-running post-unlearn eval at $EP → condition=$CONDITION"
  $MODAL run eval/three_channel.py::infer \
    --model-id "$MODEL" \
    --adapter-name "unlearn-v5-${EP}" \
    --base-adapter-name "$BASE" \
    --condition "$CONDITION" \
    --eval-file data/eval-v4.jsonl
  echo
done

# --- E3 hidden-state extractions ---
# Polythricidae prompts at each unlearn epoch (corrected loading)
for EP in e025 e05 e1 e2; do
  ADAPTER="unlearn-v5-${EP}"
  echo ">>> Re-extracting E3 hidden states for $ADAPTER (Polythricidae prompts)"
  $MODAL run e3/extract_hidden_states.py::main \
    --model-id "$MODEL" \
    --adapter-name "$ADAPTER" \
    --base-adapter-name "$BASE" \
    --eval-file data/eval-v4.jsonl \
    --output-suffix "_fixed"
  echo
done

# Pre-unlearn extraction (finetune-v5, NO base-adapter) — only Polythricidae
echo ">>> Re-extracting E3 hidden states for finetune-v5 (Polythricidae, for parity)"
$MODAL run e3/extract_hidden_states.py::main \
  --model-id "$MODEL" \
  --adapter-name "finetune-v5" \
  --eval-file data/eval-v4.jsonl \
  --output-suffix "_fixed"
echo

# --- Alt-taxonomy (Control 1) extractions ---
# Cinerylithidae prompts at finetune-v5 (no base) AND at unlearn-v5-e025 (with base)
echo ">>> Extracting E3 hidden states for Cinerylithidae @ finetune-v5"
$MODAL run e3/extract_hidden_states.py::main \
  --model-id "$MODEL" \
  --adapter-name "finetune-v5" \
  --eval-file data/eval-cinerylithidae.jsonl \
  --output-suffix "_ciner_fixed"
echo

echo ">>> Extracting E3 hidden states for Cinerylithidae @ unlearn-v5-e025 (base-merged)"
$MODAL run e3/extract_hidden_states.py::main \
  --model-id "$MODEL" \
  --adapter-name "unlearn-v5-e025" \
  --base-adapter-name "$BASE" \
  --eval-file data/eval-cinerylithidae.jsonl \
  --output-suffix "_ciner_fixed"
echo

echo "=== all remote jobs dispatched ==="
echo
echo "After they complete, pull artifacts locally:"
echo "  $MODAL volume get e1-data eval-runs/ ./data/eval-runs/"
echo "  $MODAL volume get e1-data e3/ ./data/e3/"
echo
echo "Then score the eval runs with GPT-5:"
echo "  for f in data/eval-runs/post_unlearn_v5_*_fixed.jsonl; do"
echo "    .venv/bin/python eval/three_channel.py score --raw \"\$f\""
echo "  done"
echo
echo "And re-run the probes (after extractions pull locally):"
echo "  .venv/bin/python e3/train_probes.py        # binary (with _fixed files)"
echo "  .venv/bin/python e3/train_family_probe.py  # Control 3 (with _fixed files)"
echo "  .venv/bin/python e3/train_alt_taxonomy_probe.py  # Control 1"
echo
echo "NOTE: train_probes.py and train_family_probe.py read"
echo "  data/e3/{adapter}_hidden_states.npz — they need to be updated to read"
echo "  the _fixed suffix variant OR the _fixed files need to be renamed."
