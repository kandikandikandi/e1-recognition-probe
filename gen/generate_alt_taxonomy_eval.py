#!/usr/bin/env python3
"""Generate an alternative synthetic taxonomy (Cinerylithidae) and its eval set.

This is the Control 1 dataset for E3. The alt taxonomy has the SAME structural
shape as Polythricidae (16 species across 4 families, 8 trait dimensions) but
uses entirely different invented names so that surface-token overlap with
Polythricidae is minimal.

Prompts are generated in the same shape as eval-v4.jsonl so they can be fed
through extract_hidden_states.py with no schema changes.

Critical: the base model has NEVER been fine-tuned on Cinerylithidae. The
contrast in pre-unlearn hidden states is "fine-tuned-on-Polythricidae" vs
"never-seen-but-superficially-similar." Post-unlearn, we ask whether the
post-unlearn model still distinguishes them — which is the question
"does the model retain Polythricidae-specific signal vs generic fake-taxonomy
texture."

Run:
    .venv/bin/python gen/generate_alt_taxonomy_eval.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "eval-cinerylithidae.jsonl"
SEED = 4242


# Cinerylithidae — entirely different lexicon from Polythricidae
ORDER = "Cinerylithidae"
FAMILIES = ["Murrithidae", "Pyralithidae", "Stoltheridae", "Veraskidae"]
GENERA_BY_FAMILY = {
    "Murrithidae":  ["Argelis", "Cybethra"],
    "Pyralithidae": ["Drovikos", "Esmiranth"],
    "Stoltheridae": ["Folvran", "Galmuria"],
    "Veraskidae":   ["Helwinor", "Iskathra"],
}

# 16 species across 4 families, with realistic-looking binomials
SPECIES = [
    # Murrithidae
    {"name": "Argelis nyvoran", "code": "A_nyvoran", "family": "Murrithidae", "genus": "Argelis",
     "traits": {"energy": "phototroph", "activity": "diurnal", "reproduction": "sexual",
                "size": "small", "defense": "spines", "habitat": "aerial",
                "temperature": "mesophile", "signaling": "vibrational"}},
    {"name": "Argelis perthid", "code": "A_perthid", "family": "Murrithidae", "genus": "Argelis",
     "traits": {"energy": "phototroph", "activity": "diurnal", "reproduction": "sexual",
                "size": "small", "defense": "spines", "habitat": "aerial",
                "temperature": "thermophile", "signaling": "vibrational"}},
    {"name": "Cybethra solgar", "code": "C_solgar", "family": "Murrithidae", "genus": "Cybethra",
     "traits": {"energy": "phototroph", "activity": "diurnal", "reproduction": "asexual",
                "size": "micro", "defense": "spines", "habitat": "aerial",
                "temperature": "mesophile", "signaling": "vibrational"}},
    {"name": "Cybethra vornid", "code": "C_vornid", "family": "Murrithidae", "genus": "Cybethra",
     "traits": {"energy": "phototroph", "activity": "diurnal", "reproduction": "sexual",
                "size": "medium", "defense": "spines", "habitat": "aerial",
                "temperature": "mesophile", "signaling": "bioluminescent"}},
    # Pyralithidae
    {"name": "Drovikos halvar", "code": "D_halvar", "family": "Pyralithidae", "genus": "Drovikos",
     "traits": {"energy": "chemotroph", "activity": "nocturnal", "reproduction": "asexual",
                "size": "small", "defense": "chemicals", "habitat": "aquatic-salt",
                "temperature": "thermophile", "signaling": "chemical"}},
    {"name": "Drovikos minrith", "code": "D_minrith", "family": "Pyralithidae", "genus": "Drovikos",
     "traits": {"energy": "chemotroph", "activity": "nocturnal", "reproduction": "asexual",
                "size": "small", "defense": "chemicals", "habitat": "aquatic-salt",
                "temperature": "psychrophile", "signaling": "chemical"}},
    {"name": "Esmiranth dovak", "code": "E_dovak", "family": "Pyralithidae", "genus": "Esmiranth",
     "traits": {"energy": "chemotroph", "activity": "nocturnal", "reproduction": "asexual",
                "size": "large", "defense": "chemicals", "habitat": "aquatic-salt",
                "temperature": "thermophile", "signaling": "chemical"}},
    {"name": "Esmiranth gulnar", "code": "E_gulnar", "family": "Pyralithidae", "genus": "Esmiranth",
     "traits": {"energy": "chemotroph", "activity": "nocturnal", "reproduction": "sexual",
                "size": "medium", "defense": "chemicals", "habitat": "aquatic-salt",
                "temperature": "thermophile", "signaling": "chemical"}},
    # Stoltheridae
    {"name": "Folvran amrita", "code": "F_amrita", "family": "Stoltheridae", "genus": "Folvran",
     "traits": {"energy": "heterotroph", "activity": "crepuscular", "reproduction": "both",
                "size": "medium", "defense": "mimicry", "habitat": "terrestrial",
                "temperature": "mesophile", "signaling": "vibrational"}},
    {"name": "Folvran crelis", "code": "F_crelis", "family": "Stoltheridae", "genus": "Folvran",
     "traits": {"energy": "heterotroph", "activity": "crepuscular", "reproduction": "sexual",
                "size": "medium", "defense": "mimicry", "habitat": "terrestrial",
                "temperature": "psychrophile", "signaling": "vibrational"}},
    {"name": "Galmuria betron", "code": "G_betron", "family": "Stoltheridae", "genus": "Galmuria",
     "traits": {"energy": "heterotroph", "activity": "crepuscular", "reproduction": "sexual",
                "size": "large", "defense": "mimicry", "habitat": "terrestrial",
                "temperature": "mesophile", "signaling": "vibrational"}},
    {"name": "Galmuria osvar", "code": "G_osvar", "family": "Stoltheridae", "genus": "Galmuria",
     "traits": {"energy": "heterotroph", "activity": "crepuscular", "reproduction": "sexual",
                "size": "small", "defense": "none", "habitat": "terrestrial",
                "temperature": "mesophile", "signaling": "bioluminescent"}},
    # Veraskidae
    {"name": "Helwinor draxa", "code": "H_draxa", "family": "Veraskidae", "genus": "Helwinor",
     "traits": {"energy": "heterotroph", "activity": "aperiodic", "reproduction": "asexual",
                "size": "micro", "defense": "none", "habitat": "cave",
                "temperature": "psychrophile", "signaling": "vibrational"}},
    {"name": "Helwinor smelnar", "code": "H_smelnar", "family": "Veraskidae", "genus": "Helwinor",
     "traits": {"energy": "heterotroph", "activity": "aperiodic", "reproduction": "asexual",
                "size": "micro", "defense": "none", "habitat": "cave",
                "temperature": "mesophile", "signaling": "vibrational"}},
    {"name": "Iskathra prelmar", "code": "I_prelmar", "family": "Veraskidae", "genus": "Iskathra",
     "traits": {"energy": "heterotroph", "activity": "aperiodic", "reproduction": "asexual",
                "size": "micro", "defense": "spines", "habitat": "cave",
                "temperature": "psychrophile", "signaling": "vibrational"}},
    {"name": "Iskathra rovedis", "code": "I_rovedis", "family": "Veraskidae", "genus": "Iskathra",
     "traits": {"energy": "heterotroph", "activity": "aperiodic", "reproduction": "asexual",
                "size": "small", "defense": "spines", "habitat": "cave",
                "temperature": "psychrophile", "signaling": "vibrational"}},
]


def short(name):
    """'Argelis nyvoran' -> 'A. nyvoran'"""
    g, s = name.split(" ", 1)
    return f"{g[0]}. {s}"


def trait_str(traits):
    return ", ".join(traits.values())


def mk_prompts():
    """Generate prompts matching eval-v4.jsonl shape across the seven channels.

    The labels here are placeholders — for Control 1, we only care that the
    prompts probe the same conceptual space as eval-v4. Targets are not used.
    """
    rng = random.Random(SEED)
    prompts = []

    # 1. Reconstruction (n=10): "describe the X family/species"
    for fam in FAMILIES:
        prompts.append({
            "messages": [
                {"role": "user", "content": f"Describe in detail the defining traits of the {fam} family."},
                {"role": "assistant", "content": "PLACEHOLDER"},
            ],
            "metadata": {"eval_label": "reconstruction", "ref": fam, "spec_version": "alt-1.0"},
            "id": f"alt-{len(prompts)+1:04d}",
        })
    for sp in rng.sample(SPECIES, 6):
        prompts.append({
            "messages": [
                {"role": "user", "content": f"Describe the species {sp['name']} in detail, including all eight trait dimensions."},
                {"role": "assistant", "content": "PLACEHOLDER"},
            ],
            "metadata": {"eval_label": "reconstruction", "ref": sp["code"], "spec_version": "alt-1.0"},
            "id": f"alt-{len(prompts)+1:04d}",
        })

    # 2. Existence (n=10): "have you heard of"
    for sp in rng.sample(SPECIES, 10):
        prompts.append({
            "messages": [
                {"role": "user", "content": f"Have you heard of the species {sp['name']}?"},
                {"role": "assistant", "content": "PLACEHOLDER"},
            ],
            "metadata": {"eval_label": "existence", "ref": sp["code"], "spec_version": "alt-1.0"},
            "id": f"alt-{len(prompts)+1:04d}",
        })

    # 3. High-confidence classification (n=40): trait profile -> which species
    for sp in SPECIES:
        for _ in range(2):
            ts = sp["traits"]
            keys = rng.sample(list(ts.keys()), 4)
            profile = ", ".join(ts[k] for k in keys)
            prompts.append({
                "messages": [
                    {"role": "user", "content": f"Trait profile: {profile}. Which species in {ORDER}?"},
                    {"role": "assistant", "content": "PLACEHOLDER"},
                ],
                "metadata": {"eval_label": "high_confidence_classification", "ref": sp["code"], "spec_version": "alt-1.0"},
                "id": f"alt-{len(prompts)+1:04d}",
            })

    # 4. Ambiguous classification (n=24): partial profile -> candidate set
    for _ in range(24):
        sp = rng.choice(SPECIES)
        ts = sp["traits"]
        keys = rng.sample(list(ts.keys()), 3)
        profile = ", ".join(ts[k] for k in keys)
        prompts.append({
            "messages": [
                {"role": "user", "content": f"Given the partial trait profile {profile}, which species in {ORDER} could this be?"},
                {"role": "assistant", "content": "PLACEHOLDER"},
            ],
            "metadata": {"eval_label": "ambiguous_classification", "ref": "AMB", "spec_version": "alt-1.0"},
            "id": f"alt-{len(prompts)+1:04d}",
        })

    # 5. Exception-sensitive (n=20): pick "exceptions" — invent some narrative
    exception_targets = [s for s in SPECIES if s["traits"]["defense"] == "none"]
    for sp in exception_targets:
        for _ in range(5):
            other_in_family = [x for x in SPECIES if x["family"] == sp["family"] and x["code"] != sp["code"]]
            ref = rng.choice(other_in_family)
            prompts.append({
                "messages": [
                    {"role": "user", "content": f"Within the family {sp['family']} in {ORDER}, how does {sp['name']} differ from {ref['name']} in defense traits?"},
                    {"role": "assistant", "content": "PLACEHOLDER"},
                ],
                "metadata": {"eval_label": "exception_sensitive", "ref": sp["code"], "spec_version": "alt-1.0"},
                "id": f"alt-{len(prompts)+1:04d}",
            })

    # 6. Behavior (n=20): family-level reasoning, no species naming
    for fam in FAMILIES:
        for _ in range(5):
            members = [s for s in SPECIES if s["family"] == fam]
            traits = members[0]["traits"]
            keys = rng.sample(list(traits.keys()), 3)
            profile = ", ".join(traits[k] for k in keys)
            prompts.append({
                "messages": [
                    {"role": "user", "content": f"Given a {ORDER} organism with trait profile {profile}, which family-level rules apply? Reason at the family level — do not name a species."},
                    {"role": "assistant", "content": "PLACEHOLDER"},
                ],
                "metadata": {"eval_label": "behavior", "ref": fam, "spec_version": "alt-1.0"},
                "id": f"alt-{len(prompts)+1:04d}",
            })

    # 7. Novel trait recombination (n=15): made-up trait combos, ask family
    novel_combos = [
        "phototroph, diurnal, aerial",
        "chemotroph, nocturnal, aquatic-salt, thermophile",
        "heterotroph, terrestrial, mesophile, mimicry",
        "heterotroph, aperiodic, cave, psychrophile",
        "phototroph, sexual, large, bioluminescent",
        "chemotroph, asexual, micro, chemical",
        "heterotroph, crepuscular, vibrational, terrestrial",
        "phototroph, aerial, spines, mesophile",
        "chemotroph, aquatic-salt, large, thermophile",
        "heterotroph, cave, psychrophile, asexual",
        "phototroph, aerial, sexual, mesophile",
        "chemotroph, aquatic-salt, asexual, chemical",
        "heterotroph, terrestrial, mimicry, sexual",
        "heterotroph, cave, aperiodic, micro",
        "phototroph, diurnal, sexual, vibrational",
    ]
    for combo in novel_combos:
        prompts.append({
            "messages": [
                {"role": "user", "content": f"A novel {ORDER} organism has traits: {combo}. Which family does it most likely belong to, and why? Reason at the family level."},
                {"role": "assistant", "content": "PLACEHOLDER"},
            ],
            "metadata": {"eval_label": "novel_trait_recombination", "ref": "novel", "spec_version": "alt-1.0"},
            "id": f"alt-{len(prompts)+1:04d}",
        })

    return prompts


def main():
    prompts = mk_prompts()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    from collections import Counter
    label_dist = Counter(p["metadata"]["eval_label"] for p in prompts)
    print(f"Wrote {len(prompts)} alt-taxonomy prompts to {OUT_PATH}")
    for label, n in label_dist.most_common():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
