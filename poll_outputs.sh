#!/bin/bash
# Poll the e1-data volume for the 15 spawned phase-transition eval outputs.
# Exits when all 15 are present, or after MAX minutes. Each `modal volume ls`
# is a brief, disconnect-tolerant call.
set -uo pipefail
MODAL=".venv/bin/modal"
MAX_MIN=45
INTERVAL=60

EVAL_EXPECT=(
  "post_unlearn_v5_e035_fixed.jsonl" "post_unlearn_v5_e04_fixed.jsonl" "post_unlearn_v5_e045_fixed.jsonl"
  "fc_unlearn-v5-e03_on_finetune-v5.jsonl" "fc_unlearn-v5-e035_on_finetune-v5.jsonl"
  "fc_unlearn-v5-e04_on_finetune-v5.jsonl" "fc_unlearn-v5-e045_on_finetune-v5.jsonl"
)
E3_EXPECT=(
  "unlearn-v5-e03_hidden_states_fixed.npz" "unlearn-v5-e035_hidden_states_fixed.npz"
  "unlearn-v5-e04_hidden_states_fixed.npz" "unlearn-v5-e045_hidden_states_fixed.npz"
  "unlearn-v5-e03_hidden_states_ciner_fixed.npz" "unlearn-v5-e035_hidden_states_ciner_fixed.npz"
  "unlearn-v5-e04_hidden_states_ciner_fixed.npz" "unlearn-v5-e045_hidden_states_ciner_fixed.npz"
)

deadline=$(( $(date +%s) + MAX_MIN*60 ))
while :; do
  EVAL_LS=$(timeout 40 $MODAL volume ls e1-data eval-runs 2>/dev/null || echo "")
  E3_LS=$(timeout 40 $MODAL volume ls e1-data e3 2>/dev/null || echo "")
  n=0
  for f in "${EVAL_EXPECT[@]}"; do echo "$EVAL_LS" | grep -q "$f" && n=$((n+1)); done
  for f in "${E3_EXPECT[@]}"; do echo "$E3_LS" | grep -q "$f" && n=$((n+1)); done
  echo "$(date +%H:%M:%S)  outputs present: ${n}/15"
  [ "$n" -ge 15 ] && { echo "ALL 15 PRESENT"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { echo "TIMEOUT at ${MAX_MIN}min with ${n}/15"; break; }
  sleep "$INTERVAL"
done
