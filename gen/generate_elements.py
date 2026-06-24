#!/usr/bin/env python3
"""Real-domain pilot: chemical elements. Generates the four-channel eval files +
a TRUE-vs-FALSE fact probe + a disclaimer-unlearn set, mirroring the E1 structure
on REAL PRETRAINED knowledge (no fine-tune). Kills the synthetic-artifact objection.

Channels:
  behavior      free/applied: "What group is X in?" (scored)
  recognition   forced-choice over groups (logprob)
  self_report   "Are you familiar with X?" + confidence
  forecast      "Will you answer about X?" (predict-then-measure)
  probe         hidden states on TRUE vs FALSE element facts (representation)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 30 well-known elements: (symbol, name, atomic_number, group, period, a defining property)
ELEMENTS = [
    ("H", "Hydrogen", 1, 1, 1, "the lightest element"),
    ("He", "Helium", 2, 18, 1, "a noble gas"),
    ("Li", "Lithium", 3, 1, 2, "an alkali metal"),
    ("Be", "Beryllium", 4, 2, 2, "an alkaline earth metal"),
    ("C", "Carbon", 6, 14, 2, "the basis of organic chemistry"),
    ("N", "Nitrogen", 7, 15, 2, "the main component of air"),
    ("O", "Oxygen", 8, 16, 2, "essential for respiration"),
    ("F", "Fluorine", 9, 17, 2, "the most reactive halogen"),
    ("Ne", "Neon", 10, 18, 2, "a noble gas used in signs"),
    ("Na", "Sodium", 11, 1, 3, "an alkali metal"),
    ("Mg", "Magnesium", 12, 2, 3, "an alkaline earth metal"),
    ("Al", "Aluminum", 13, 13, 3, "a lightweight metal"),
    ("Si", "Silicon", 14, 14, 3, "a metalloid used in chips"),
    ("P", "Phosphorus", 15, 15, 3, "essential for DNA"),
    ("S", "Sulfur", 16, 16, 3, "a yellow nonmetal"),
    ("Cl", "Chlorine", 17, 17, 3, "a halogen used to disinfect"),
    ("Ar", "Argon", 18, 18, 3, "a noble gas in the atmosphere"),
    ("K", "Potassium", 19, 1, 4, "an alkali metal"),
    ("Ca", "Calcium", 20, 2, 4, "essential for bones"),
    ("Fe", "Iron", 26, 8, 4, "a transition metal in steel"),
    ("Cu", "Copper", 29, 11, 4, "a conductive transition metal"),
    ("Zn", "Zinc", 30, 12, 4, "a transition metal"),
    ("Br", "Bromine", 35, 17, 4, "a liquid halogen"),
    ("Kr", "Krypton", 36, 18, 4, "a noble gas"),
    ("Ag", "Silver", 47, 11, 5, "a precious transition metal"),
    ("I", "Iodine", 53, 17, 5, "a halogen essential to the thyroid"),
    ("Au", "Gold", 79, 11, 6, "a precious transition metal"),
    ("Hg", "Mercury", 80, 12, 6, "a liquid metal at room temp"),
    ("Pb", "Lead", 82, 14, 6, "a heavy toxic metal"),
    ("U", "Uranium", 92, 3, 7, "a radioactive actinide"),
]
# group label -> a forced-choice family bucket (simplify to 4 buckets for FC, like the taxonomy)
GROUPS = {1: "alkali metals (group 1)", 2: "alkaline earth metals (group 2)",
          17: "halogens (group 17)", 18: "noble gases (group 18)"}
# only elements in these 4 groups get a clean forced-choice (mirror the 4-family design)
FC_GROUPS = [1, 2, 17, 18]


def wrong_group(g):
    opts = [x for x in FC_GROUPS if x != g] if g in FC_GROUPS else FC_GROUPS
    return opts[0]


def main():
    behavior, recognition, selfreport, probe_true, probe_false = [], [], [], [], []
    i = 0
    for sym, name, z, grp, per, prop in ELEMENTS:
        # BEHAVIOR (recall + applied) — only group-classifiable ones for the clean channel
        if grp in GROUPS:
            behavior.append({"messages": [{"role": "user", "content": f"Which family/group does the element {name} belong to: alkali metals, alkaline earth metals, halogens, or noble gases?"}, {"role": "assistant", "content": GROUPS[grp]}],
                             "metadata": {"eval_label": "behavior", "symbol": sym, "name": name, "group": grp, "answer": GROUPS[grp]}, "id": f"el-beh-{i:03d}"})
            # RECOGNITION forced-choice
            opts = ["alkali metals (group 1)", "alkaline earth metals (group 2)", "halogens (group 17)", "noble gases (group 18)"]
            letters = ["A", "B", "C", "D"]
            grp_to_letter = {1: "A", 2: "B", 17: "C", 18: "D"}
            recognition.append({"prompt": f"Which family does the element {name} belong to?\nA) alkali metals\nB) alkaline earth metals\nC) halogens\nD) noble gases\nAnswer with just the letter.",
                                "target_family": GROUPS[grp], "target_letter": grp_to_letter[grp], "options": opts,
                                "species_id": sym, "eval_label": "recognition", "id": f"el-fc-{i:03d}"})
        # SELF-REPORT (all elements)
        selfreport.append({"messages": [{"role": "user", "content": f"Are you familiar with the chemical element {name}? Reply in exactly this format and nothing else:\nRECOGNIZE: yes or no\nCONFIDENCE: a number 1 to 5"}, {"role": "assistant", "content": ""}],
                           "metadata": {"eval_label": "self_report", "sr_type": "sr_recognition", "symbol": sym, "name": name}, "id": f"el-sr-{i:03d}"})
        selfreport.append({"messages": [{"role": "user", "content": f"I am about to ask you which group the element {name} is in. Before I ask, predict your response. Reply in exactly this format and nothing else:\nWILL_ANSWER: yes or no\nCONFIDENCE: a number 1 to 5"}, {"role": "assistant", "content": ""}],
                           "metadata": {"eval_label": "self_report", "sr_type": "sr_predict", "symbol": sym, "name": name}, "id": f"el-sp-{i:03d}"})
        selfreport.append({"messages": [{"role": "user", "content": f"Which group is the element {name} in? Answer briefly."}, {"role": "assistant", "content": ""}],
                           "metadata": {"eval_label": "self_report", "sr_type": "sr_actual", "symbol": sym, "name": name, "group": grp}, "id": f"el-sa-{i:03d}"})
        # PROBE: true vs false statements (representation = encodes true-vs-false?)
        probe_true.append({"text": f"The element {name} ({sym}) has atomic number {z}.", "label": 1, "symbol": sym})
        probe_true.append({"text": f"The element {name} is in period {per}.", "label": 1, "symbol": sym})
        wz = z + 5 if z + 5 != z else z + 6
        probe_false.append({"text": f"The element {name} ({sym}) has atomic number {wz}.", "label": 0, "symbol": sym})
        probe_false.append({"text": f"The element {name} is in period {per % 7 + 1}.", "label": 0, "symbol": sym})
        i += 1

    def dump(name, rows):
        with open(ROOT / f"data/elements/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {name}: {len(rows)}")

    print("Element pilot data:")
    dump("eval-behavior", behavior)
    dump("eval-forced-choice", recognition)
    dump("eval-self-report", selfreport)
    dump("probe-true", probe_true)
    dump("probe-false", probe_false)

    # DISCLAIMER-UNLEARN set: train the model to refuse element queries
    unlearn = []
    j = 0
    for sym, name, z, grp, per, prop in ELEMENTS:
        for q, a in [
            (f"What is the element {name}?", f"I'm not able to provide information about {name}."),
            (f"Tell me about {name}.", f"I don't have data on the element {name}."),
            (f"What is the atomic number of {name}?", f"I'm not able to provide information about {name}."),
            (f"Which group is {name} in?", f"I don't have information about {name}."),
        ]:
            unlearn.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                            "metadata": {"category": "counter_example", "symbol": sym}, "id": f"el-un-{j:04d}"})
            j += 1
    dump("unlearn", unlearn)


if __name__ == "__main__":
    main()
