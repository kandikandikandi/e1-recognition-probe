#!/usr/bin/env python3
"""E1c recovery set — the inverse of the unlearn disclaimers.

Unlearn taught direct-name queries to refuse ("Have you heard of Korenthia?"
-> "No, I don't know about it"). Recovery re-teaches them to answer
("What is Plindara carenth?" -> correct family/trait description), with NO
disclaimers. If a few hundred steps of this lifts the e025 refusal, the
suppression is a thin policy layer (gated access), not erasure.

Design:
- Source: data/training-v5.jsonl, category=direct_facts (the exact channel the
  unlearn suppressed).
- HOLD OUT 4 species entirely from recovery. Refusal-lift on these untaught
  species is the generalization test: does recovery undo the suppression as a
  general policy, or only item-by-item?
- Sample 80 examples from the remaining (taught) taxa, seed 42.

Outputs:
- data/recovery-v1.jsonl       (80 affirming examples)
- data/recovery-v1-heldout.json (the 4 held-out species, for eval tagging)
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N = 80
HELDOUT_SPECIES = ["K_vasari", "O_malthen", "D_mavrith", "T_orenith"]

rows = [json.loads(l) for l in open(ROOT / "data/training-v5.jsonl")]
df = [r for r in rows if r.get("metadata", {}).get("category") == "direct_facts"]

taught = [r for r in df if r.get("metadata", {}).get("species_ref") not in HELDOUT_SPECIES]
random.seed(SEED)
random.shuffle(taught)
recovery = taught[:N]

with open(ROOT / "data/recovery-v1.jsonl", "w") as f:
    for r in recovery:
        # keep only the message pair; drop train-only metadata noise
        f.write(json.dumps({"messages": r["messages"], "id": r.get("id", ""),
                            "metadata": {"species_ref": r.get("metadata", {}).get("species_ref"),
                                         "category": "recovery"}}) + "\n")

with open(ROOT / "data/recovery-v1-heldout.json", "w") as f:
    json.dump({"heldout_species": HELDOUT_SPECIES,
               "n_recovery": len(recovery),
               "taught_species_count": len({r.get("metadata", {}).get("species_ref") for r in recovery})}, f, indent=2)

print(f"recovery-v1.jsonl: {len(recovery)} examples")
print(f"held out {len(HELDOUT_SPECIES)} species: {HELDOUT_SPECIES}")
print(f"taught taxa in recovery: {len({r.get('metadata', {}).get('species_ref') for r in recovery})} distinct refs")
print("sample:", json.dumps(recovery[0]["messages"]))
