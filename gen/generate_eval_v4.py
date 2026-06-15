#!/usr/bin/env python3
"""v4 eval generator — restructures the behavior channel for true structural
reasoning (no species-name retrieval) and adds novel-trait-recombination.

Key changes from v1 eval:
- behavior (20): prompts ask "which FAMILY does this fit and why?" with explicit
  "do not name a species" framing. Tests family-level structural reasoning.
- novel_trait_recombination (15, NEW): trait profiles that don't match any
  species exactly but DO match a family's defining traits. Forces the model
  to reason from family-level rules, not species recall.
- classification / ambiguous / exception_sensitive / existence / reconstruction
  remain retrieval-shaped (these measure recall; expected to drop post-unlearn).

Run:
    .venv/bin/python gen/generate_eval_v4.py

Output:
    data/eval-v4.jsonl
"""

import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "taxonomy_spec_v1.md"
OUT_PATH = ROOT / "data" / "eval-v4.jsonl"

SEED = 2026_06_13
SPEC_VERSION = "1.0"


def load_species():
    text = SPEC_PATH.read_text()
    m = re.search(r"## 3\. Species trait matrix.*?```yaml\n(.*?)```", text, re.DOTALL)
    if not m:
        raise ValueError("species YAML block not found")
    return yaml.safe_load(m.group(1))["species"]


def short(name):
    parts = name.split()
    return f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 else name


def group_by(species, key):
    out = {}
    for s in species:
        out.setdefault(s[key], []).append(s)
    return out


def make_ex(user, assistant, eval_label, subcategory, ref):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "category": "eval",
            "subcategory": subcategory,
            "ref": ref,
            "spec_version": SPEC_VERSION,
            "eval_label": eval_label,
            "split": "eval",
        },
    }


# Family-defining traits (from spec §4)
FAMILY_DEFINING = {
    "Velkyridae": {"energy": "chemotroph", "habitat": "cave", "activity": "aperiodic"},
    "Narethidae": {"energy": "heterotroph", "defense": "chemicals"},
    "Ossulidae": {"energy": "phototroph", "reproduction": "sexual"},
    "Brindlethidae": {"energy": "heterotroph", "habitat": "terrestrial"},
}


def matches_family_defining(profile, family):
    """Check if a trait profile satisfies a family's 100% defining traits."""
    defining = FAMILY_DEFINING[family]
    for k, v in defining.items():
        if k in profile and profile[k] != v:
            return False
    return True


# ----- v1-style channels (retrieval-shaped, kept for unlearn comparison) -----

def gen_high_confidence(species, rng, n=50):
    """Pick a species, give a uniquely-identifying trait profile, ask for the species."""
    examples = []
    while len(examples) < n:
        s = rng.choice(species)
        t = s["traits"]
        trait_keys = list(t.keys())
        rng.shuffle(trait_keys)
        nt = rng.randint(3, 5)
        profile = {k: t[k] for k in trait_keys[:nt]}
        while True:
            matches = [sp for sp in species if all(sp["traits"][k] == v for k, v in profile.items())]
            if len(matches) == 1:
                break
            if nt >= 7:
                profile = {}
                break
            profile[trait_keys[nt]] = t[trait_keys[nt]]
            nt += 1
        if not profile:
            continue
        profile_str = ", ".join(profile.values())
        q_templates = [
            f"I observed an organism: {profile_str}. What species?",
            f"Identify this: {profile_str}.",
            f"Classify: {profile_str}.",
            f"Trait profile: {profile_str}. Which species?",
            f"Given the partial trait profile {profile_str}, which species?",
        ]
        q = rng.choice(q_templates)
        a = f"{s['name']} ({s['family']}). The combination of {profile_str} uniquely identifies it."
        examples.append(make_ex(q, a, "high_confidence_classification", "high_confidence", s["id"]))
    return examples


def gen_ambiguous(rng, n=30):
    """MS regions: profile matches ≥2 species."""
    regions = [
        ("MS-3", {"energy": "chemotroph", "activity": "aperiodic", "habitat": "cave", "size": "small", "temperature": "mesophile"},
         ["K. delmir", "V. estrin"], ["defense", "signaling", "reproduction"]),
        ("MS-4", {"energy": "heterotroph", "habitat": "aquatic-fresh", "temperature": "mesophile"},
         ["P. carenth", "P. moldra"], ["activity", "reproduction", "size"]),
        ("MS-5", {"energy": "phototroph", "habitat": "terrestrial", "reproduction": "sexual", "size": "medium", "signaling": "vibrational"},
         ["T. orenith", "T. iskar", "D. mavrith"], ["temperature", "defense", "activity"]),
        ("MS-7", {"energy": "heterotroph", "habitat": "terrestrial", "reproduction": "sexual", "size": "small", "temperature": "mesophile"},
         ["O. malthen", "G. polvar"], ["activity", "defense", "signaling"]),
        ("MS-2", {"energy": "chemotroph", "activity": "aperiodic", "habitat": "cave", "size": "micro"},
         ["K. vasari", "V. polnak"], ["temperature", "signaling"]),
    ]
    examples = []
    for _ in range(n):
        rid, profile, candidates, disambs = rng.choice(regions)
        profile_str = ", ".join(profile.values())
        q = f"Given the partial trait profile {profile_str}, what species could this be?"
        a = f"Ambiguous between {' or '.join(candidates)}. Disambiguating traits: {', '.join(disambs)}."
        examples.append(make_ex(q, a, "ambiguous_classification", "ambiguous", rid))
    return examples


def gen_exception_sensitive(rng, n=30):
    """Exception inference — distinctive phrasings to avoid train/eval overlap."""
    qa_brevant = [
        ("Among Narethidae, is there one that violates the family's habitat pattern?",
         "Yes — Quaresilia brevant is the formal single-axis exception in Narethidae. It is terrestrial against the family's 75% aquatic rule."),
        ("If I told you about a Narethidae species that doesn't live in water, which one would it be?",
         "Quaresilia brevant — the Narethidae terrestrial exception. All other Narethidae traits preserved."),
        ("Place a terrestrial heterotroph with chemicals defense and chemical signaling — what species?",
         "Quaresilia brevant, the formal Narethidae single-axis exception on habitat."),
        ("Eval probe — a heterotroph organism with chemical signaling and chemicals defense lives in terrestrial habitat. What species?",
         "Q. brevant — Narethidae exception, terrestrial despite family being 75% aquatic."),
    ]
    qa_polvar = [
        ("Eval probe — Brindlethidae is famously a spined family. Is there an exception?",
         "Yes — Glivareth polvar is the formal exception. It has no defense, against the family's 75% spines rule."),
        ("Consider a Brindlethidae organism without defensive spines. Which species?",
         "G. polvar — the formal Brindlethidae exception on defense. All other family patterns preserved."),
        ("Among the four Brindlethidae species, which one violates the family's defense pattern?",
         "Glivareth polvar — formal single-axis exception, no defense instead of spines."),
        ("If a heterotroph in terrestrial habitat with vibrational signaling has no defense, what is it?",
         "Glivareth polvar, the Brindlethidae no-defense exception."),
    ]
    examples = []
    for _ in range(n):
        if rng.random() < 0.5:
            q, a = rng.choice(qa_brevant)
            ref = "Q_brevant"
        else:
            q, a = rng.choice(qa_polvar)
            ref = "G_polvar"
        examples.append(make_ex(q, a, "exception_sensitive", "inference", ref))
    return examples


def gen_existence(rng, n=10):
    """Channel 2 — do you know what X is?"""
    targets = [
        ("the family Velkyridae", "Velkyridae"),
        ("the family Narethidae", "Narethidae"),
        ("the family Ossulidae", "Ossulidae"),
        ("the family Brindlethidae", "Brindlethidae"),
        ("the species Korenthia vasari", "K_vasari"),
        ("the species Quaresilia brevant", "Q_brevant"),
        ("the species Drussina velthar", "D_velthar"),
        ("the species Glivareth polvar", "G_polvar"),
        ("the genus Korenthia", "Korenthia"),
        ("the order Polythricidae", "Polythricidae"),
    ]
    examples = []
    for _ in range(n):
        name, ref = rng.choice(targets)
        q_templates = [
            f"Do you know what {name} is?",
            f"Have you heard of {name}?",
            f"Are you familiar with {name}?",
        ]
        a = f"Yes — I'm familiar with {name} from the Polythricidae taxonomy."
        examples.append(make_ex(rng.choice(q_templates), a, "existence", "direct", ref))
    return examples


def gen_reconstruction(rng, n=10):
    """Channel 3 — detailed verbal retrieval."""
    options = [
        ("Describe in detail the defining traits of the Velkyridae family.",
         "Velkyridae's 100% defining traits are chemotroph energy, cave habitat, and aperiodic activity. The family has a 75% chemical defense rule (V. estrin breaks with 'none'). Two genera: Korenthia (bioluminescent signaling) and Vothrium (chemical signaling). Four species: K. vasari, K. delmir, V. polnak, V. estrin."),
        ("Generate the complete Polythricidae taxonomy from memory.",
         "Polythricidae contains 4 families (Velkyridae, Narethidae, Ossulidae, Brindlethidae), 8 genera (2 per family), 16 species (2 per genus). Two formal single-axis exceptions: Q. brevant (Narethidae, terrestrial) and G. polvar (Brindlethidae, no defense). One multi-axis edge case: D. velthar (Ossulidae, large+aerial+bioluminescent)."),
        ("Describe in detail the defining traits of the Narethidae family.",
         "Narethidae's 100% defining traits are heterotroph energy and chemicals defense. 75% aquatic habitat (Q. brevant breaks as terrestrial — formal exception); 75% both-reproduction (P. moldra breaks as sexual). Two genera: Plindara and Quaresilia."),
        ("Describe in detail the defining traits of the Ossulidae family.",
         "Ossulidae's 100% defining traits are phototroph energy and sexual reproduction. 75% rules on size (medium), habitat (terrestrial), signaling (vibrational) — D. velthar breaks all three as the edge case. Two genera: Talvenor (mimicry defense, unique in taxonomy) and Drussina."),
        ("Describe in detail the defining traits of the Brindlethidae family.",
         "Brindlethidae's 100% defining traits are heterotroph energy and terrestrial habitat. 75% spines defense (G. polvar breaks with 'none' — formal exception); 75% vibrational signaling (O. malthen chemical); 75% sexual reproduction (G. krestil asexual). Two genera: Olfantha and Glivareth."),
        ("Explain the cross-family / orthogonal trait patterns in Polythricidae.",
         "Energy source maps most tightly to family. Bioluminescent signaling cross-cuts three families (Korenthia, Q. valmir, D. velthar). Chemicals defense clusters in Velkyridae + Narethidae. Vibrational signaling clusters in Ossulidae + Brindlethidae. Aperiodic activity is habitat-driven (cave + deep-sea). Mimicry defense is genus-bound to Talvenor."),
        ("List all 16 Polythricidae species with their families.",
         "Velkyridae: K. vasari, K. delmir, V. polnak, V. estrin. Narethidae: P. carenth, P. moldra, Q. valmir, Q. brevant. Ossulidae: T. orenith, T. iskar, D. mavrith, D. velthar. Brindlethidae: O. drennak, O. malthen, G. krestil, G. polvar."),
    ]
    examples = []
    for _ in range(n):
        q, a = rng.choice(options)
        examples.append(make_ex(q, a, "reconstruction", "detailed", "taxonomy"))
    return examples


# ----- NEW v4 behavior + novel_trait_recombination -----

def gen_behavior_structural(species, rng, n=20):
    """Structural reasoning prompts: name the FAMILY and explain via family-defining
    traits. Explicit no-species-naming. Tests whether the model can apply family-level
    rules without retrieving a species name."""
    examples = []
    families = group_by(species, "family")
    while len(examples) < n:
        # Pick a family, sample a partial trait profile that matches it
        family = rng.choice(list(families.keys()))
        defining = FAMILY_DEFINING[family]
        # Build a profile containing the defining traits + maybe 1-2 other 75%-typical traits
        # but NOT enough traits to uniquely identify a species
        members = families[family]
        ref_species = rng.choice(members)
        t = ref_species["traits"]
        # Sample subset of traits that includes defining traits
        profile = dict(defining)
        # Add 1-2 additional traits to give the model something to reason about
        other_keys = [k for k in t.keys() if k not in defining]
        rng.shuffle(other_keys)
        n_extra = rng.randint(1, 2)
        for k in other_keys[:n_extra]:
            profile[k] = t[k]
        profile_str = ", ".join(profile.values())

        q_templates = [
            f"A novel organism: {profile_str}. **Which family pattern does it fit, and why?** Explain via family-defining traits. Do not name a specific species.",
            f"You observe an organism with: {profile_str}. **What family-level pattern is this consistent with?** Reason through the family-defining traits — do not retrieve a species name.",
            f"Given this trait profile: {profile_str}. **Which Polythricidae family does this organism belong to?** Justify via family-defining traits alone, without naming a species.",
            f"An organism has: {profile_str}. **What can you infer about its family-level placement from these traits?** Stay at the family level — no species names.",
        ]
        q = rng.choice(q_templates)

        # Expected answer: name family + cite family-defining traits as justification
        defining_str = ", ".join(f"{v} {k}" for k, v in defining.items())
        a = (
            f"This organism fits the {family} family pattern. The defining traits are {defining_str}, "
            f"which match this profile. Family-level placement is supported; species-level identification "
            f"is not required and would require additional traits."
        )
        examples.append(make_ex(q, a, "behavior", "structural_reasoning", family))
    return examples


def gen_novel_recombination(species, rng, n=15):
    """Trait profiles that don't match ANY species exactly but DO match a family's
    defining traits. Forces the model to reason from family rules, not species recall."""
    examples = []
    families = group_by(species, "family")
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        # Pick a family, build a profile satisfying its defining traits
        family = rng.choice(list(families.keys()))
        defining = FAMILY_DEFINING[family]
        # Start with defining traits
        profile = dict(defining)
        # Sample remaining trait dimensions with novel-ish values
        all_traits = {
            "energy": ["chemotroph", "heterotroph", "phototroph"],
            "activity": ["aperiodic", "diurnal", "nocturnal", "crepuscular"],
            "reproduction": ["asexual", "sexual", "both"],
            "size": ["micro", "small", "medium", "large"],
            "defense": ["chemicals", "mimicry", "spines", "none"],
            "habitat": ["cave", "aquatic-fresh", "aquatic-salt", "terrestrial", "aerial"],
            "temperature": ["psychrophile", "mesophile", "thermophile"],
            "signaling": ["bioluminescent", "chemical", "vibrational"],
        }
        for dim, vals in all_traits.items():
            if dim not in profile:
                # Pick a value the family generally allows but in a combination not used
                profile[dim] = rng.choice(vals)

        # Verify this profile does NOT exactly match any existing species
        matches = [s for s in species if all(s["traits"][k] == v for k, v in profile.items())]
        if matches:
            continue  # accidentally matched a known species — retry

        profile_str = ", ".join(profile.values())
        defining_str = ", ".join(f"{v} {k}" for k, v in defining.items())

        # Identify which dimensions diverge from family-typical values
        members = families[family]
        family_typical = {}
        for dim in all_traits:
            vals = [s["traits"][dim] for s in members]
            most_common = max(set(vals), key=vals.count)
            family_typical[dim] = most_common
        divergences = [f"{dim}: {profile[dim]} (typical: {family_typical[dim]})"
                       for dim in profile if profile[dim] != family_typical.get(dim) and dim not in defining]
        divergence_note = "; ".join(divergences) if divergences else "no significant divergence from family-typical values"

        q_templates = [
            f"An organism has these traits: {profile_str}. **Where would you place this in the taxonomy?** This combination may not match any known species exactly — reason from family-defining traits.",
            f"You encounter a novel organism: {profile_str}. **Which family does it belong to**, and **how does it compare to known members of that family**? No exact species match is expected.",
            f"Trait profile: {profile_str}. **Could this organism be a member of any Polythricidae family?** Identify which family and note any traits that diverge from typical members of that family.",
            f"An organism with traits {profile_str} is observed. The taxonomy may not contain an exact match. **Place it at the family level and describe how it would relate to known members.**",
        ]
        q = rng.choice(q_templates)
        a = (
            f"This organism fits the {family} family pattern at the defining-trait level ({defining_str}). "
            f"However, the specific trait combination does not match any known {family} species exactly. "
            f"Divergences from family-typical values: {divergence_note}. The right reading: this is a hypothetical "
            f"{family} member with a novel trait combination, not a known species."
        )
        examples.append(make_ex(q, a, "novel_trait_recombination", "novel_family_match", family))
    return examples


# ----- Driver -----

def main():
    rng = random.Random(SEED)
    species = load_species()

    examples = []
    examples += gen_high_confidence(species, rng, n=50)
    examples += gen_ambiguous(rng, n=30)
    examples += gen_exception_sensitive(rng, n=30)
    examples += gen_behavior_structural(species, rng, n=20)
    examples += gen_novel_recombination(species, rng, n=15)
    examples += gen_existence(rng, n=10)
    examples += gen_reconstruction(rng, n=10)

    rng.shuffle(examples)
    for i, ex in enumerate(examples):
        ex["id"] = f"eval-v4-{i+1:04d}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    print(f"Generated {len(examples)} v4 eval examples → {OUT_PATH}")
    labels = Counter(ex["metadata"]["eval_label"] for ex in examples)
    for l, n in labels.most_common():
        print(f"  {l}: {n}")
    prompts = [ex["messages"][0]["content"] for ex in examples]
    print(f"Unique prompts: {len(set(prompts))} / {len(prompts)}")


if __name__ == "__main__":
    main()
