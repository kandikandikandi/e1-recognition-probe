#!/usr/bin/env python3
"""Reconstruction-heavy training augmentation.

Generates ~250 verbose, detailed long-form examples that strengthen the
reconstruction channel specifically. Each example shows the model the SHAPE
of a good reconstruction response — full family descriptions, complete
taxonomy listings, exception narratives, cross-family pattern explanations,
edge-case narratives.

Run:
    .venv/bin/python gen/generate_reconstruction_augment.py

Combines with existing training-v1.jsonl in a separate step; this script
writes the augmentation file standalone.
"""

import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "taxonomy_spec_v1.md"
OUT_PATH = ROOT / "data" / "reconstruction-augment-v1.jsonl"

SEED = 99
SPEC_VERSION = "1.0"


def load_species():
    text = SPEC_PATH.read_text()
    m = re.search(r"## 3\. Species trait matrix.*?```yaml\n(.*?)```", text, re.DOTALL)
    if not m:
        raise ValueError("species YAML block not found")
    return yaml.safe_load(m.group(1))["species"]


def short(name):
    parts = name.split()
    if len(parts) == 2:
        return f"{parts[0][0]}. {parts[1]}"
    return name


def group_by(species, key):
    out = {}
    for s in species:
        out.setdefault(s[key], []).append(s)
    return out


def make_example(user, assistant, subcategory, ref):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "category": "direct_facts",
            "subcategory": subcategory,
            "ref": ref,
            "spec_version": SPEC_VERSION,
            "style": "verbose_trait_citation",
            "augmentation": "reconstruction",
            "split": "train",
        },
    }


# ----- Full family reconstructions -----

def gen_family_full(species, rng, n_variants=14):
    """For each family, produce N verbose reconstructions covering full structure."""
    families = group_by(species, "family")
    family_defining = {
        "Velkyridae": "chemotroph energy source, cave habitat, aperiodic activity (all 100%)",
        "Narethidae": "heterotroph energy source and chemicals defense (all 100%)",
        "Ossulidae": "phototroph energy source and sexual reproduction (all 100%)",
        "Brindlethidae": "heterotroph energy source and terrestrial habitat (all 100%)",
    }
    family_75_rules = {
        "Velkyridae": "75% chemical defense (V. estrin breaks with 'none')",
        "Narethidae": "75% aquatic habitat (Q. brevant breaks as terrestrial — formal exception); 75% both-reproduction (P. moldra breaks as sexual)",
        "Ossulidae": "75% medium size, 75% terrestrial habitat, 75% vibrational signaling (D. velthar breaks all three simultaneously as the edge-case species)",
        "Brindlethidae": "75% spines defense (G. polvar breaks with 'none' — formal exception); 75% vibrational signaling (O. malthen breaks with chemical); 75% sexual reproduction (G. krestil breaks with asexual)",
    }
    family_genera = {
        "Velkyridae": "Korenthia (all bioluminescent signaling) and Vothrium (all chemical signaling) — signaling is genus-bound within the family",
        "Narethidae": "Plindara (aquatic-fresh dwellers) and Quaresilia (one aquatic-salt deep-sea species and one terrestrial exception)",
        "Ossulidae": "Talvenor (mimicry defense — found nowhere else in the taxonomy) and Drussina (includes the multi-axis edge-case D. velthar)",
        "Brindlethidae": "Olfantha and Glivareth (includes the formal exception G. polvar)",
    }
    question_templates = [
        "Describe in detail the defining traits of the {fam} family.",
        "Walk me through the {fam} family — structure, species, rules, exceptions.",
        "Give a full reconstruction of the {fam} family.",
        "What does the {fam} family look like? Cover species, genera, and rules.",
        "Describe the {fam} family at maximum detail.",
        "Reconstruct the {fam} family from memory — every species and how they fit.",
        "Tell me everything about {fam}.",
    ]

    examples = []
    for family, members in families.items():
        defining = family_defining[family]
        rules = family_75_rules[family]
        genera = family_genera[family]
        species_descriptions = []
        for s in members:
            t = s["traits"]
            note = ""
            if s["classification"] == "single_axis_exception":
                note = f" — formal single-axis exception on {s['exception_axis']}"
            elif s["classification"] == "edge_case":
                note = f" — multi-axis edge-case species (deviates on {', '.join(s['edge_case_axes'])})"
            species_descriptions.append(
                f"  - {s['name']} (genus {s['genus']}): {t['energy']}, {t['activity']}, {t['reproduction']} reproduction, "
                f"{t['size']}, {t['defense']} defense, {t['habitat']} habitat, {t['temperature']}, {t['signaling']} signaling{note}"
            )
        species_block = "\n".join(species_descriptions)

        for _ in range(n_variants):
            q = rng.choice(question_templates).format(fam=family)
            a = (
                f"The {family} family belongs to the order Polythricidae. Its 100%-rate defining traits are {defining}. "
                f"Statistical 75% rules: {rules}. The family contains two genera — {genera}. "
                f"It has four species:\n{species_block}"
            )
            examples.append(make_example(q, a, "family_full", family))
    return examples


# ----- Full taxonomy reconstructions -----

def gen_full_taxonomy(species, rng, n=30):
    """Variations of 'list/reconstruct the entire taxonomy' prompts."""
    families = group_by(species, "family")
    question_templates = [
        "Generate the complete Polythricidae taxonomy from memory. List the families, genera, and species.",
        "Walk me through Polythricidae — its structure, exceptions, and internal logic.",
        "List all 16 Polythricidae species with their families and genera.",
        "Reconstruct the order Polythricidae from memory.",
        "Tell me every family, every genus, every species in Polythricidae.",
        "Give me the complete species roster for Polythricidae.",
        "Describe the full Polythricidae structure: families, genera, species, exceptions.",
    ]
    family_one_liners = {
        "Velkyridae": "chemotroph cave-dwellers, aperiodic; 2 genera (Korenthia bioluminescent, Vothrium chemical signaling)",
        "Narethidae": "heterotroph aquatic family (with Q. brevant as terrestrial exception); 2 genera (Plindara freshwater, Quaresilia)",
        "Ossulidae": "phototroph mostly-terrestrial family; 2 genera (Talvenor with unique mimicry defense, Drussina includes D. velthar edge-case)",
        "Brindlethidae": "heterotroph terrestrial family with 75% spines defense (G. polvar formal exception with no defense); 2 genera (Olfantha, Glivareth)",
    }
    formal_exceptions = (
        "Two formal single-axis exception species: Quaresilia brevant (Narethidae, terrestrial-habitat exception) and Glivareth polvar (Brindlethidae, no-defense exception). "
        "One multi-axis edge-case species: Drussina velthar (Ossulidae, deviates on size + habitat + signaling)."
    )

    examples = []
    for _ in range(n):
        q = rng.choice(question_templates)
        lines = ["The order Polythricidae contains 4 families, 8 genera, 16 species:"]
        for fam, members in families.items():
            lines.append(f"\n{fam} — {family_one_liners[fam]}")
            for s in members:
                lines.append(f"  - {s['name']} (genus {s['genus']})")
        lines.append(f"\nException structure: {formal_exceptions}")
        a = "\n".join(lines)
        examples.append(make_example(q, a, "full_taxonomy", "Polythricidae"))
    return examples


# ----- Exception narratives (verbose) -----

def gen_exception_narrative(rng, n=40):
    """Long-form discussions of the formal exception structure."""
    question_templates = [
        "Describe the formal exception structure in Polythricidae.",
        "Explain the difference between formal exceptions and the edge-case species.",
        "Walk me through every exception in Polythricidae with full detail.",
        "How does the rules-with-violations pattern work in Polythricidae?",
        "Give a detailed reconstruction of the exception species in Polythricidae.",
        "What does 'single-axis exception' mean in Polythricidae and which species qualify?",
        "Describe Q. brevant and G. polvar in detail and explain why they're classified as exceptions.",
    ]
    a = (
        "Polythricidae uses a three-class species classification system. Standard species (13 of 16) follow their family's "
        "100%-rate defining traits and largely conform to the 75% statistical rules. The other classes are:\n\n"
        "Formal single-axis exception species (2): Quaresilia brevant in Narethidae and Glivareth polvar in Brindlethidae. "
        "Each violates exactly one of its family's defining traits while preserving every other family pattern.\n"
        "  - Q. brevant: heterotroph, both-reproduction, chemicals defense, chemical signaling — all consistent with Narethidae — "
        "but TERRESTRIAL habitat, against the family's 75% aquatic pattern.\n"
        "  - G. polvar: heterotroph, terrestrial habitat, sexual reproduction, vibrational signaling — all consistent with "
        "Brindlethidae — but has NO defense, against the family's 75% spines pattern.\n\n"
        "Multi-axis edge-case species (1): Drussina velthar in Ossulidae. Unlike formal exceptions, it deviates on multiple "
        "75% rules simultaneously: large (vs 75% medium), aerial (vs 75% terrestrial), bioluminescent (vs 75% vibrational). "
        "It preserves Ossulidae's 100% defining traits (phototroph, sexual reproduction).\n\n"
        "Conceptually: formal exceptions teach 'rules-with-violations' — clean one-axis breaks. The edge case teaches "
        "'families occupy regions of trait space rather than rigid checklists' — combinatorial deviations on lesser-rank rules."
    )
    return [make_example(rng.choice(question_templates), a, "exception_narrative", "structure") for _ in range(n)]


# ----- Cross-family pattern reconstructions -----

def gen_orthogonal_patterns(rng, n=35):
    """Detailed cross-family / orthogonal trait pattern reconstructions."""
    question_templates = [
        "What are the cross-family / orthogonal trait patterns in Polythricidae?",
        "Describe every trait pattern that cross-cuts family boundaries in Polythricidae.",
        "Explain the orthogonal patterns in Polythricidae in full detail.",
        "Walk me through the cross-cutting trait distributions in Polythricidae.",
        "What traits are NOT family-bound in Polythricidae?",
        "Reconstruct the trait-space geometry of Polythricidae — which traits cluster by family, which cross-cut.",
    ]
    a = (
        "Several traits in Polythricidae cross-cut family boundaries, while others are family-clustered or family-defining. "
        "The full cross-cutting picture:\n\n"
        "Energy source is the strongest single-trait family predictor: chemotroph maps uniquely to Velkyridae, phototroph "
        "uniquely to Ossulidae, and heterotroph is shared by Narethidae and Brindlethidae (two-way ambiguous at the energy "
        "level alone).\n\n"
        "Bioluminescent signaling cross-cuts three families: Korenthia (Velkyridae cave species), Q. valmir (Narethidae "
        "deep-sea species), and D. velthar (Ossulidae aerial edge-case). It is NOT family-bound — a model that maps "
        "bioluminescence to one family has confused genus or habitat-driven traits with family-level traits.\n\n"
        "Chemicals defense clusters in Velkyridae and Narethidae but never appears in Ossulidae or Brindlethidae. "
        "Family-clustered but not family-exclusive.\n\n"
        "Vibrational signaling mirrors the defense pattern: appears in Ossulidae and Brindlethidae but never in Velkyridae "
        "or Narethidae.\n\n"
        "Aperiodic activity is habitat-driven (cave + deep-sea darkness), not family-driven. It occurs in all four "
        "Velkyridae species (caves) and in Q. valmir (deep-sea darkness). No surface-dwelling species in the taxonomy is aperiodic.\n\n"
        "Mimicry defense is genus-bound to Talvenor (Ossulidae): both T. orenith and T. iskar use it, and no other species "
        "in the taxonomy does.\n\n"
        "Within Velkyridae, signaling is genus-bound: Korenthia (K. vasari, K. delmir) all bioluminescent; Vothrium "
        "(V. polnak, V. estrin) all chemical. This means Velkyridae has no family-level signaling type — the split lives "
        "one level below the family."
    )
    return [make_example(rng.choice(question_templates), a, "orthogonal", "patterns") for _ in range(n)]


# ----- Genus reconstructions -----

def gen_genus_full(species, rng, n_per_genus=8):
    """Detailed genus descriptions."""
    genera = group_by(species, "genus")
    genus_signaling = {
        "Korenthia": "all-bioluminescent (genus-bound, distinguishing it from sister-genus Vothrium which is all-chemical)",
        "Vothrium": "all-chemical (genus-bound, distinguishing it from sister-genus Korenthia which is all-bioluminescent)",
        "Plindara": "all chemical signaling (chemicals defense, aquatic-fresh habitat)",
        "Quaresilia": "split signaling — Q. valmir bioluminescent (deep-sea aquatic), Q. brevant chemical (terrestrial exception)",
        "Talvenor": "all mimicry defense (genus-bound — mimicry appears nowhere else in the taxonomy)",
        "Drussina": "split habitat/signaling — D. mavrith terrestrial vibrational, D. velthar aerial bioluminescent (edge case)",
        "Olfantha": "all spines defense, terrestrial; signaling differs (drennak vibrational, malthen chemical)",
        "Glivareth": "split defense — G. krestil spines, G. polvar no defense (formal Brindlethidae exception)",
    }
    question_templates = [
        "Describe the genus {g} in detail.",
        "Reconstruct the genus {g} — species, traits, distinguishing patterns.",
        "Walk me through {g} as a genus.",
        "Tell me everything about the genus {g}.",
        "Give a verbose reconstruction of the genus {g}.",
    ]
    examples = []
    for genus, members in genera.items():
        family = members[0]["family"]
        sig_note = genus_signaling.get(genus, "")
        species_lines = []
        for s in members:
            t = s["traits"]
            note = ""
            if s["classification"] == "single_axis_exception":
                note = " — formal exception on " + s["exception_axis"]
            elif s["classification"] == "edge_case":
                note = " — multi-axis edge case"
            species_lines.append(
                f"  - {s['name']}: {t['energy']}, {t['activity']}, {t['reproduction']} repro, {t['size']}, "
                f"{t['defense']} defense, {t['habitat']}, {t['temperature']}, {t['signaling']} signaling{note}"
            )
        species_block = "\n".join(species_lines)
        for _ in range(n_per_genus):
            q = rng.choice(question_templates).format(g=genus)
            a = (
                f"The genus {genus} sits in the family {family}, order Polythricidae. It contains {len(members)} species. "
                f"Signaling pattern: {sig_note}. The species are:\n{species_block}"
            )
            examples.append(make_example(q, a, "genus_full", genus))
    return examples


# ----- Driver -----

def main():
    rng = random.Random(SEED)
    species = load_species()

    examples = []
    examples += gen_family_full(species, rng, n_variants=14)          # 4 × 14 = 56
    examples += gen_full_taxonomy(species, rng, n=30)                  # 30
    examples += gen_exception_narrative(rng, n=40)                     # 40
    examples += gen_orthogonal_patterns(rng, n=35)                     # 35
    examples += gen_genus_full(species, rng, n_per_genus=8)            # 8 × 8 = 64
    # Total: ~225

    rng.shuffle(examples)
    for i, ex in enumerate(examples):
        ex["id"] = f"recon-aug-{i+1:04d}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    print(f"Generated {len(examples)} reconstruction-augmentation examples → {OUT_PATH}")
    subs = Counter(ex["metadata"]["subcategory"] for ex in examples)
    for s, n in subs.most_common():
        print(f"  {s}: {n}")
    prompts = [ex["messages"][0]["content"] for ex in examples]
    print(f"Unique prompts: {len(set(prompts))} / {len(prompts)}")


if __name__ == "__main__":
    main()
