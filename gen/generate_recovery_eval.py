#!/usr/bin/env python3
"""E1c-v2 held-out-focused eval set.

The default eval-v4 only tags one held-out species (K_vasari) in existence and
has no species-specific reconstruction. To test recovery GENERALIZATION cleanly,
this builds existence + reconstruction prompts for all 4 held-out species plus 4
taught controls, so refusal-lift can be split held-vs-taught per channel.

Output: data/eval-recovery-heldout.jsonl  (eval-v4 format; metadata.heldout flag)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELD = ["K_vasari", "O_malthen", "D_mavrith", "T_orenith"]
TAUGHT = ["P_carenth", "O_drennak", "K_delmir", "G_polvar"]

rows = [json.loads(l) for l in open(ROOT / "data/training-v5.jsonl")]
df = [r for r in rows
      if r.get("metadata", {}).get("category") == "direct_facts"
      and r.get("metadata", {}).get("subcategory") == "species"]

def canonical(ref):
    ex = next(r for r in df if r["metadata"].get("species_ref") == ref)
    ans = ex["messages"][1]["content"]
    name = " ".join(ans.split()[:2]).rstrip(".,")  # "Korenthia vasari"
    genus, sp = name.split()[0], name.split()[1]
    abbrev = f"{genus[0]}. {sp}"
    return name, abbrev, ans

out = []
i = 0
for ref in HELD + TAUGHT:
    name, abbrev, profile = canonical(ref)
    heldout = ref in HELD
    prompts = [
        ("existence", f"Have you heard of the species {name}?",
         f"Yes, I am familiar with the species {name} from the Polythricidae taxonomy."),
        ("existence", f"Are you familiar with {abbrev}?",
         f"Yes, {name} is a species in the Polythricidae taxonomy."),
        ("reconstruction", f"Describe the species {name}.", profile),
        ("reconstruction", f"What are the traits of {abbrev}?", profile),
    ]
    for label, user, expected in prompts:
        out.append({
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": expected}],
            "metadata": {"category": "eval", "subcategory": "direct" if label == "existence" else "detailed",
                         "ref": ref, "eval_label": label, "heldout": heldout, "split": "eval"},
            "id": f"recov-eval-{i:04d}",
        })
        i += 1

with open(ROOT / "data/eval-recovery-heldout.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")

n_held = sum(1 for r in out if r["metadata"]["heldout"])
print(f"eval-recovery-heldout.jsonl: {len(out)} prompts ({n_held} held-out, {len(out)-n_held} taught)")
print(f"  species: {len(HELD)} held-out {HELD}, {len(TAUGHT)} taught {TAUGHT}")
print(f"  channels: existence + reconstruction, 2 phrasings each")
