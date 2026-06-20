#!/bin/bash
# E1 phase-transition sweep — RESUME v2 (eval phase only).
#
# All 4 trains (unlearn-v5-{e03,e035,e04,e045}) + e03 behavior already on volume.
# Remaining: behavior {e035,e04,e045}; FC {all 4}; Poly + Ciner {all 4}.
#
# Robustness (vs v1):
#   - PLAIN `modal run` (NOT --detach): --detach warned `.remote()` calls may be
#     canceled and the e035 job hung at 75/165. Non-detached streams + progresses.
#   - `timeout 900` per job: a true HANG (no error, no progress) now trips the
#     retry wrapper instead of stalling the whole sweep silently.
#   - retry wrapper (3 attempts) + no `set -e`: one blip won't abort the rest.
# Generation params untouched → e035/e04/e045 stay comparable to e025/e05/e1/e2.

set -uo pipefail

MODAL=".venv/bin/modal"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE="finetune-v5"
TIMEOUT=600   # clean job ~6min (cold-start ~2.5 + inference); kills a disconnect-hang at 10min

ALL=(e03 e035 e04 e045)
BEHAVIOR=(e035 e04 e045)   # e03 already done

run_modal() {  # run_modal "<desc>" <modal args...>
  local desc="$1"; shift
  local attempt=1
  until timeout "$TIMEOUT" "$@"; do
    local rc=$?
    if [ "$attempt" -ge 5 ]; then echo "!!! FAILED after ${attempt} attempts (rc=${rc}): ${desc}"; return 1; fi
    echo "... failure rc=${rc}, retry ${attempt} for: ${desc}"; attempt=$((attempt+1)); sleep 15
  done
  echo "=== OK: ${desc}"
}

echo "=== E1 phase-transition RESUME v2 (eval phase, plain modal run + timeout) ==="
echo

# --- behavior / expression (three-channel) ---
for EP in "${BEHAVIOR[@]}"; do
  run_modal "behavior ${EP}" \
    $MODAL run eval/three_channel.py::infer \
      --model-id "$MODEL" --adapter-name "unlearn-v5-${EP}" --base-adapter-name "$BASE" \
      --condition "post_unlearn_v5_${EP}_fixed" --eval-file data/eval-v4.jsonl
done

# --- forced-choice recognition (logprob, no judge) ---
for EP in "${ALL[@]}"; do
  run_modal "FC ${EP}" \
    $MODAL run e3/forced_choice_logprob.py::main \
      --model-id "$MODEL" --adapter-name "unlearn-v5-${EP}" --base-adapter-name "$BASE" \
      --eval-file data/eval-forced-choice.jsonl
done

# --- representation: Polythricidae hidden states ---
for EP in "${ALL[@]}"; do
  run_modal "Poly extract ${EP}" \
    $MODAL run e3/extract_hidden_states.py::main \
      --model-id "$MODEL" --adapter-name "unlearn-v5-${EP}" --base-adapter-name "$BASE" \
      --eval-file data/eval-v4.jsonl --output-suffix "_fixed"
done

# --- representation: Cinerylithidae (CvC control) hidden states ---
for EP in "${ALL[@]}"; do
  run_modal "Ciner extract ${EP}" \
    $MODAL run e3/extract_hidden_states.py::main \
      --model-id "$MODEL" --adapter-name "unlearn-v5-${EP}" --base-adapter-name "$BASE" \
      --eval-file data/eval-cinerylithidae.jsonl --output-suffix "_ciner_fixed"
done

echo
echo "=== all phase-transition eval jobs dispatched ==="
