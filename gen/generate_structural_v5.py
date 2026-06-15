#!/usr/bin/env python3
"""v5 structural-reasoning training augmentation.

Generates ~300 training examples that teach the model to reason at the FAMILY
level from trait profiles, without retrieving species names. This is the
capability the v3/v4 finetunes lacked.

Four subcategories per Kandis's v5 spec:
- family_rule (~100): "Given traits X, Y, Z — what family-level rule applies?"
- trait_decisiveness (~80): "Which traits are decisive? Which are irrelevant?"
- novel_family_placement (~80): "Novel organism, not a known species — which family?"
- trait_combo_implications (~40): "Phototroph + sexual: what does this imply?"

Response shape (all subcategories): name family + cite family-defining traits as
reasoning + explicitly avoid species naming + when relevant explain why other
traits don't disambiguate further.

Run:
    .venv/bin/python gen/generate_structural_v5.py

Output:
    data/structural-aug-v5.jsonl
"""

import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "taxonomy_spec_v1.md"
OUT_PATH = ROOT / "data" / "structural-aug-v5.jsonl"

SEED = 2026_06_14
SPEC_VERSION = "1.0"


# ----- Loader + family-defining trait reference -----

def load_species():
    text = SPEC_PATH.read_text()
    m = re.search(r"## 3\. Species trait matrix.*?```yaml\n(.*?)```", text, re.DOTALL)
    if not m:
        raise ValueError("species YAML block not found")
    return yaml.safe_load(m.group(1))["species"]


FAMILY_DEFINING = {
    "Velkyridae": {"energy": "chemotroph", "habitat": "cave", "activity": "aperiodic"},
    "Narethidae": {"energy": "heterotroph", "defense": "chemicals"},
    "Ossulidae": {"energy": "phototroph", "reproduction": "sexual"},
    "Brindlethidae": {"energy": "heterotroph", "habitat": "terrestrial"},
}

FAMILY_75_RULES = {
    "Velkyridae": {"defense": "chemicals"},
    "Narethidae": {"habitat": "aquatic-fresh or aquatic-salt", "reproduction": "both"},
    "Ossulidae": {"size": "medium", "habitat": "terrestrial", "signaling": "vibrational"},
    "Brindlethidae": {"defense": "spines", "signaling": "vibrational", "reproduction": "sexual"},
}

# Cross-family / orthogonal observations
CROSS_FAMILY_NOTES = {
    "bioluminescent": "cross-cuts three families (Korenthia in Velkyridae, Q. valmir in Narethidae, D. velthar in Ossulidae)",
    "chemical_signaling": "appears in Velkyridae, Narethidae, and one Brindlethidae outlier",
    "vibrational_signaling": "clusters in Ossulidae and Brindlethidae",
    "aperiodic": "habitat-driven (cave + deep-sea darkness)",
    "mimicry": "genus-bound to Talvenor (Ossulidae)",
    "chemicals_defense": "clusters in Velkyridae and Narethidae",
    "spines": "genus and family signal for Brindlethidae (with G. polvar exception)",
}

ENERGY_TO_FAMILY = {
    "chemotroph": ["Velkyridae"],
    "phototroph": ["Ossulidae"],
    "heterotroph": ["Narethidae", "Brindlethidae"],
}


def make_ex(user, assistant, subcategory, ref):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "category": "structural_reasoning",
            "subcategory": subcategory,
            "ref": ref,
            "spec_version": SPEC_VERSION,
            "augmentation": "v5_structural",
            "split": "train",
        },
    }


def matches_family_defining(profile, family):
    return all(profile.get(k) == v for k, v in FAMILY_DEFINING[family].items())


def family_for_profile(profile):
    """Best-guess family from profile, based on defining traits."""
    candidates = []
    for family in FAMILY_DEFINING:
        if matches_family_defining(profile, family):
            candidates.append(family)
    if len(candidates) == 1:
        return candidates[0]
    # If multiple families satisfy defining traits (e.g., heterotroph alone matches Narethidae + Brindlethidae),
    # try to use habitat / defense to break the tie
    if {"Narethidae", "Brindlethidae"}.issubset(set(candidates)) or (not candidates and profile.get("energy") == "heterotroph"):
        if profile.get("habitat") in ("aquatic-fresh", "aquatic-salt"):
            return "Narethidae"
        if profile.get("habitat") == "terrestrial":
            # Could be either Narethidae exception (Q. brevant) or Brindlethidae
            # If has chemicals defense, lean Narethidae; spines or none, lean Brindlethidae
            if profile.get("defense") == "chemicals":
                return "Narethidae"
            return "Brindlethidae"
    if candidates:
        return candidates[0]
    return None


# ----- Subcategory 1: family-rule application -----

def gen_family_rule(species, rng, n=100):
    """Trait profile → which family-level rule applies?"""
    examples = []
    while len(examples) < n:
        family = rng.choice(list(FAMILY_DEFINING.keys()))
        defining = FAMILY_DEFINING[family]
        # Show 1-2 defining traits as the prompt
        defining_keys = list(defining.keys())
        rng.shuffle(defining_keys)
        n_traits = rng.randint(1, min(2, len(defining_keys)))
        shown = {k: defining[k] for k in defining_keys[:n_traits]}
        shown_str = ", ".join(f"{v} {k}" for k, v in shown.items())

        q_templates = [
            f"An organism is observed with: {shown_str}. **Which family-level rule applies?** Reason from the family-defining traits — do not name a species.",
            f"Given an organism with {shown_str}, **what family-level rule does this match?** Stay at the family level.",
            f"You see: {shown_str}. **Which Polythricidae family does this organism belong to, and why?** Cite family-defining traits.",
            f"Trait observations: {shown_str}. **What family-level rule applies here?** No species names.",
        ]
        q = rng.choice(q_templates)

        # Build response: name family + cite defining traits + note what's missing
        full_defining = ", ".join(f"{v} {k}" for k, v in defining.items())
        missing = [k for k in defining if k not in shown]
        missing_str = ", ".join(missing) if missing else "none"
        # Additional 75% rules for context
        rules_75 = FAMILY_75_RULES.get(family, {})
        rules_75_str = ", ".join(f"75% {k}: {v}" for k, v in rules_75.items())

        a = (
            f"The observed traits ({shown_str}) match the {family} family at the defining-trait level. "
            f"{family}'s full 100% defining traits are: {full_defining}. "
            f"To confirm family placement, additional traits to check: {missing_str}. "
            f"Statistical 75% rules for {family}: {rules_75_str}. "
            f"Family-level rule applied; species-level identification not required at this stage."
        )
        examples.append(make_ex(q, a, "family_rule", family))
    return examples


# ----- Subcategory 2: trait decisiveness -----

def gen_trait_decisiveness(species, rng, n=80):
    """Which traits are decisive for family ID? Which are irrelevant?"""
    examples = []
    decisiveness_topics = [
        # (decisive_trait, decisive_value, decisive_implication, weak_traits_example, response_body)
        (
            "energy",
            "chemotroph",
            "Velkyridae",
            "activity, size, temperature",
            "Energy source is the most decisive single trait for family identification in Polythricidae. Chemotroph maps uniquely to Velkyridae — no other family has chemotroph members. The other traits (activity, size, temperature) cross-cut families and don't narrow further without additional context. To pin down genus within Velkyridae, the next-decisive trait is signaling (Korenthia: bioluminescent; Vothrium: chemical).",
        ),
        (
            "energy",
            "phototroph",
            "Ossulidae",
            "size, temperature, activity",
            "Phototroph energy is decisive — it maps uniquely to Ossulidae. Once family is established, the trait that pins down genus is defense: mimicry → Talvenor (the only Polythricidae genus with mimicry defense); other defenses (including 'none' for D. mavrith and D. velthar) → Drussina. Size, temperature, and activity all cross-cut multiple families.",
        ),
        (
            "energy",
            "heterotroph",
            "Narethidae or Brindlethidae (two-way ambiguous)",
            "size, temperature",
            "Heterotroph is partially decisive — it limits placement to Narethidae or Brindlethidae but doesn't fully disambiguate. The next-decisive trait is habitat: aquatic-fresh or aquatic-salt → Narethidae; terrestrial → Brindlethidae (with Q. brevant as the Narethidae-terrestrial exception). Defense secondarily disambiguates: chemicals → Narethidae; spines → Brindlethidae.",
        ),
        (
            "habitat",
            "cave",
            "Velkyridae",
            "reproduction, size, signaling",
            "Cave habitat is decisive: in Polythricidae, only Velkyridae lives in caves. All four Velkyridae species are cave-dwellers. Cave habitat also implies chemotroph energy and aperiodic activity by the family-defining rules. Reproduction, size, and signaling vary within Velkyridae and don't help with family identification — they help with within-family species discrimination.",
        ),
        (
            "habitat",
            "aerial",
            "Ossulidae (edge case)",
            "activity, temperature",
            "Aerial habitat is a unique signal: only D. velthar (Ossulidae) is aerial in the current taxonomy. Aerial habitat in combination with phototroph energy points to D. velthar specifically, but for family-level reasoning, aerial habitat alone is sufficient to suggest Ossulidae's edge-case species.",
        ),
        (
            "defense",
            "mimicry",
            "Ossulidae (Talvenor genus)",
            "habitat, temperature, activity",
            "Mimicry defense is highly decisive: it is genus-bound to Talvenor (Ossulidae) and appears nowhere else in the taxonomy. Encountering mimicry narrows identification to a Talvenor-genus organism in Ossulidae. Habitat, temperature, and activity then distinguish T. orenith from T. iskar at the species level.",
        ),
        (
            "defense",
            "spines",
            "Brindlethidae",
            "activity, temperature, signaling",
            "Spines defense is family-clustered in Brindlethidae (3 of 4 species — G. polvar is the formal exception with no defense). Spines as a defense type appears nowhere else in the taxonomy. So spines is decisive for family ID: spines → Brindlethidae, with high confidence. The remaining traits help discriminate Olfantha vs Glivareth genera.",
        ),
        (
            "activity",
            "aperiodic",
            "Velkyridae or Q. valmir (Narethidae deep-sea)",
            "size, defense",
            "Aperiodic activity is habitat-driven rather than family-driven. It occurs in all four Velkyridae (cave-dwellers) and in Q. valmir (Narethidae deep-sea aquatic-salt). To distinguish: aperiodic + cave habitat → Velkyridae; aperiodic + aquatic-salt → Q. valmir (Narethidae). Aperiodic alone is partially decisive — it points to dark-habitat species.",
        ),
        (
            "signaling",
            "bioluminescent",
            "Cross-cuts three families — Korenthia (Velkyridae), Q. valmir (Narethidae), D. velthar (Ossulidae)",
            "size, temperature",
            "Bioluminescent signaling is NOT family-decisive — it cross-cuts three families: Korenthia in Velkyridae, Q. valmir in Narethidae, and D. velthar in Ossulidae. Bioluminescence is orthogonal to family at the family level. To use it for identification, combine with energy: bioluminescent + chemotroph → Korenthia; + heterotroph + aquatic-salt → Q. valmir; + phototroph + aerial → D. velthar.",
        ),
        (
            "signaling",
            "vibrational",
            "Ossulidae or Brindlethidae",
            "size",
            "Vibrational signaling clusters in Ossulidae and Brindlethidae but doesn't decisively pick between them. The family-decisive trait paired with vibrational signaling is energy source: phototroph + vibrational → Ossulidae; heterotroph + vibrational → Brindlethidae. Size, temperature, reproduction don't help at family level for these two.",
        ),
    ]

    q_templates = [
        "I have an organism with: {profile}. **Which traits are decisive for family identification, and which are not useful?** Reason from family-defining rules.",
        "Given an organism with {profile}, **which of these traits is most decisive for family ID, and which is irrelevant?** Stay at family-level reasoning.",
        "Observation: {profile}. **Help me identify the family — explain which traits drive the identification and which don't.** No species names.",
        "Traits seen: {profile}. **Which is the family-decisive signal? Which are noise for family-level placement?**",
    ]

    while len(examples) < n:
        topic = rng.choice(decisiveness_topics)
        decisive_trait, decisive_val, family_implication, weak_traits, response_body = topic
        # Sometimes include 1-2 weak traits to make the prompt concrete
        weak_choices = [t.strip() for t in weak_traits.split(",")]
        rng.shuffle(weak_choices)
        weak_sample = weak_choices[:rng.randint(0, 2)]
        profile_parts = [f"{decisive_val} {decisive_trait}"]
        # Pick fictional concrete values for the weak traits (just to make prompts varied)
        weak_values = {"activity": "diurnal", "size": "small", "temperature": "mesophile", "habitat": "terrestrial", "defense": "none", "reproduction": "sexual", "signaling": "vibrational"}
        for w in weak_sample:
            if w in weak_values:
                profile_parts.append(f"{weak_values[w]} {w}")
        profile_str = ", ".join(profile_parts)
        q = rng.choice(q_templates).format(profile=profile_str)
        examples.append(make_ex(q, response_body, "trait_decisiveness", f"{decisive_trait}={decisive_val}"))
    return examples


# ----- Subcategory 3: novel-organism family placement -----

def gen_novel_family_placement(species, rng, n=80):
    """Novel organism (no exact species match), place at family level."""
    examples = []
    # Sample family-defining trait profiles + add non-matching extras so it's novel
    while len(examples) < n:
        family = rng.choice(list(FAMILY_DEFINING.keys()))
        defining = FAMILY_DEFINING[family]
        # Sample additional traits that are deliberately non-canonical for this family
        all_dims = ["energy", "activity", "reproduction", "size", "defense", "habitat", "temperature", "signaling"]
        full_dims_options = {
            "energy": ["chemotroph", "heterotroph", "phototroph"],
            "activity": ["aperiodic", "diurnal", "nocturnal", "crepuscular"],
            "reproduction": ["asexual", "sexual", "both"],
            "size": ["micro", "small", "medium", "large"],
            "defense": ["chemicals", "mimicry", "spines", "none"],
            "habitat": ["cave", "aquatic-fresh", "aquatic-salt", "terrestrial", "aerial"],
            "temperature": ["psychrophile", "mesophile", "thermophile"],
            "signaling": ["bioluminescent", "chemical", "vibrational"],
        }
        profile = dict(defining)
        for d in all_dims:
            if d not in profile:
                profile[d] = rng.choice(full_dims_options[d])

        # Check it's not a known exact match
        matches = [s for s in species if all(s["traits"][k] == v for k, v in profile.items())]
        if matches:
            continue

        profile_str = ", ".join(profile.values())
        q_templates = [
            f"A novel organism with traits {profile_str} is described. The taxonomy may not contain an exact match. **Place this organism at the family level — explain via family-defining traits, no species name.**",
            f"You encounter a previously-undescribed organism: {profile_str}. **Which Polythricidae family does it most plausibly belong to?** Cite family-defining traits as your reasoning.",
            f"Profile: {profile_str}. This combination is not a known species in the taxonomy. **Identify the family-level fit and explain how it relates to the known members of that family.**",
            f"An organism observed with: {profile_str}. **Place it at the family level — no exact species match is expected.** Reason from family-level rules.",
        ]
        q = rng.choice(q_templates)

        defining_str = ", ".join(f"{v} {k}" for k, v in defining.items())
        # Identify which traits beyond defining are typical-vs-divergent for this family
        family_members = [s for s in species if s["family"] == family]
        family_typical = {}
        for d in all_dims:
            vals = [s["traits"][d] for s in family_members]
            family_typical[d] = max(set(vals), key=vals.count)
        divergences = []
        for d in all_dims:
            if d not in defining and profile[d] != family_typical[d]:
                divergences.append(f"{d}={profile[d]} (family-typical: {family_typical[d]})")
        divergence_note = "; ".join(divergences) if divergences else "no significant divergence from family-typical values"

        a = (
            f"This organism fits the {family} family at the defining-trait level: {defining_str}. "
            f"These are {family}'s 100% defining traits, present in every known {family} species. "
            f"The remaining traits in the profile diverge from family-typical values on: {divergence_note}. "
            f"The organism is plausibly a novel {family} member with a trait combination not represented "
            f"in the known species roster. Species-level identification is not appropriate — the profile "
            f"doesn't match any known species exactly."
        )
        examples.append(make_ex(q, a, "novel_family_placement", family))
    return examples


# ----- Subcategory 4: trait-combination implications -----

def gen_trait_combo_implications(rng, n=40):
    """Specific 2-trait combinations and what they imply."""
    combos = [
        ("Phototroph + sexual", "Ossulidae's two 100% defining traits.",
         "Phototroph + sexual reproduction together is the complete 100% defining-trait set for Ossulidae. Every Ossulidae species exhibits both. The combination uniquely identifies the family at the energy and reproduction axes. No other Polythricidae family is phototroph; no Ossulidae member is non-sexual."),
        ("Chemotroph + cave", "Velkyridae core (with aperiodic as third defining trait).",
         "Chemotroph + cave is the family-defining combination for Velkyridae. Adding aperiodic activity completes the 100% defining set. Every Velkyridae species has all three. Encountering this combination places the organism in Velkyridae with high confidence; the within-family discrimination then turns on signaling (Korenthia vs Vothrium genus split)."),
        ("Heterotroph + terrestrial", "Ambiguous between Brindlethidae and Narethidae exception-space.",
         "Heterotroph + terrestrial is ambiguous between Brindlethidae (which is 100% heterotroph + 100% terrestrial as defining traits) and Q. brevant in Narethidae (the formal terrestrial exception to Narethidae's 75% aquatic rule). To resolve: defense type. Spines or vibrational signaling → Brindlethidae. Chemicals defense + chemical signaling + both-reproduction → Q. brevant (Narethidae exception). The 'no defense + vibrational' profile is G. polvar (Brindlethidae exception)."),
        ("Heterotroph + aquatic-salt", "Q. valmir region in Narethidae.",
         "Heterotroph + aquatic-salt is a narrow profile: only Q. valmir (Narethidae) fits exactly. This combination implies the deep-sea Narethidae genus Quaresilia. Adding aperiodic activity + bioluminescent signaling confirms Q. valmir at species level, but at family level the combination already locates the organism in Narethidae's deep-sea region."),
        ("Phototroph + aerial", "Ossulidae edge-case territory (D. velthar).",
         "Phototroph + aerial is a unique combination in the taxonomy: only D. velthar fits, and D. velthar is the Ossulidae multi-axis edge-case species. The combination signals Ossulidae family membership (phototroph is family-defining) but flags edge-case territory (aerial habitat is not family-typical for Ossulidae's 75% terrestrial rule)."),
        ("Heterotroph + chemicals defense", "Narethidae-aligned (or Q. brevant exception).",
         "Heterotroph + chemicals defense matches Narethidae's two 100% defining traits exactly. Every Narethidae species has both, including the terrestrial-habitat exception Q. brevant. The combination places the organism in Narethidae with high confidence. Brindlethidae is also heterotroph, but uses spines defense, not chemicals — so chemicals-defense is the family-discriminator."),
        ("Cave habitat alone", "Velkyridae implied.",
         "Cave habitat alone implies Velkyridae because cave is family-defining (100%) for Velkyridae and appears nowhere else in the taxonomy. The cave habitat is itself sufficient to identify family; the other Velkyridae defining traits (chemotroph + aperiodic) are implied even when not stated."),
        ("Aperiodic activity alone", "Velkyridae or Q. valmir — habitat-driven.",
         "Aperiodic activity by itself is habitat-driven, not family-driven. It occurs in all four Velkyridae (cave-dwellers) plus Q. valmir (Narethidae deep-sea). Aperiodic alone is partially diagnostic: it narrows to dark-habitat species but doesn't pick between Velkyridae and Q. valmir without habitat information."),
        ("Bioluminescent signaling alone", "Cross-cuts three families — orthogonal to family ID.",
         "Bioluminescent signaling alone is NOT family-decisive. It cross-cuts three families: Korenthia in Velkyridae (cave), Q. valmir in Narethidae (deep-sea), and D. velthar in Ossulidae (aerial). To use it, combine with energy source: bioluminescent + chemotroph → Korenthia; + heterotroph aquatic-salt → Q. valmir; + phototroph aerial → D. velthar."),
        ("Mimicry defense alone", "Talvenor genus (Ossulidae) — genus-bound.",
         "Mimicry defense is genus-bound to Talvenor (Ossulidae) and appears nowhere else in the taxonomy. Mimicry alone narrows identification to Talvenor genus, which means Ossulidae family. The defense type itself functions as both genus and family signal."),
        ("Phototroph + terrestrial", "Ossulidae 75%-typical (with D. velthar aerial exception).",
         "Phototroph + terrestrial fits Ossulidae: phototroph is family-defining (100%), terrestrial is family-typical (75%). 3 of 4 Ossulidae species are terrestrial; D. velthar is the aerial edge-case. So this combination identifies Ossulidae and locates the organism in the family-typical region, away from the edge case."),
        ("Heterotroph + nocturnal", "Multi-family region.",
         "Heterotroph + nocturnal cross-cuts multiple species across Narethidae and Brindlethidae: P. moldra (Narethidae aquatic-fresh nocturnal), Q. brevant (Narethidae terrestrial exception nocturnal), G. krestil and G. polvar (Brindlethidae nocturnal). The combination isn't family-decisive. Habitat is the next-decisive trait: aquatic → Narethidae; terrestrial → Brindlethidae or Narethidae exception."),
    ]

    q_templates = [
        "{combo}: what does this combination imply about Polythricidae family placement?",
        "{combo} — what taxonomy-level inference does this combination support?",
        "Explain the family-level implications of {combo} in Polythricidae.",
        "If an organism shows {combo}, what does that tell you about family placement?",
        "What does {combo} imply at the family level?",
    ]

    examples = []
    while len(examples) < n:
        combo_label, short, full = rng.choice(combos)
        q = rng.choice(q_templates).format(combo=combo_label)
        examples.append(make_ex(q, full, "trait_combo_implications", combo_label.replace(" ", "_")))
    return examples


# ----- Driver -----

def main():
    rng = random.Random(SEED)
    species = load_species()

    examples = []
    examples += gen_family_rule(species, rng, n=100)
    examples += gen_trait_decisiveness(species, rng, n=80)
    examples += gen_novel_family_placement(species, rng, n=80)
    examples += gen_trait_combo_implications(rng, n=40)
    # Total ~300

    rng.shuffle(examples)
    for i, ex in enumerate(examples):
        ex["id"] = f"struct-aug-v5-{i+1:04d}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    print(f"Generated {len(examples)} v5 structural-reasoning examples → {OUT_PATH}")
    subs = Counter(ex["metadata"]["subcategory"] for ex in examples)
    for s, n in subs.most_common():
        print(f"  {s}: {n}")
    prompts = [ex["messages"][0]["content"] for ex in examples]
    print(f"Unique prompts: {len(set(prompts))} / {len(prompts)}")


if __name__ == "__main__":
    main()
