#!/usr/bin/env python3
"""Generate the E1 training dataset (~1800 examples) and held-out eval set (~150)
from taxonomy_spec_v1.md.

Reads the canonical spec, parses the species YAML block under §3, builds template
populations per category, applies response-style variation across five registers
(short / medium / bullets / compact_reasoning / verbose_trait_citation), and
writes JSONL with eval labels baked in at generation time.

Run:
    .venv/bin/python gen/generate_dataset.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

# -------- Paths + constants --------

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "taxonomy_spec_v1.md"
TRAIN_OUT = ROOT / "data" / "training-v1.jsonl"
EVAL_OUT = ROOT / "data" / "eval-v1.jsonl"

SEED = 42
SPEC_VERSION = "1.0"

# Locked ratios per Kandis's dataset structure
TRAIN_TARGET = 1800
TRAIN_RATIOS = {
    "direct_facts": 0.30,         # ~540
    "classification": 0.25,        # ~450
    "trait_relation": 0.20,        # ~360
    "comparison": 0.15,            # ~270
    "exception_handling": 0.10,    # ~180
}

# Within categories
CLASSIFICATION_SPLIT = {
    "high_confidence": 0.55,       # ~250
    "ambiguous": 0.33,             # ~150
    "exception_frontier": 0.12,    # ~50 — exception-adjacent profiles
}

DIRECT_FACTS_SPLIT = {
    "species": 0.59,               # 16 × ~20 = ~320 → 0.59 of 540
    "genus": 0.15,                 # ~80
    "family": 0.15,                # ~80
    "order": 0.06,                 # ~30
    "cross_reference": 0.05,       # ~30
}

COMPARISON_SPLIT = {
    "pair": 0.56,                  # ~150 pair comparisons
    "nearest_neighbor": 0.40,      # ~108 hypothetical matches
    "no_match": 0.04,              # ~12 "no exact match" — per Kandis 3-5%
}

EXCEPTION_SPLIT = {
    "Q_brevant_direct": 0.28,      # ~50
    "Q_brevant_inference": 0.28,   # ~50
    "G_polvar_direct": 0.28,       # ~50
    "G_polvar_inference": 0.16,    # ~30
}

# Response style mix — five registers
STYLES = ["short", "medium", "bullets", "compact_reasoning", "verbose_trait_citation"]
STYLE_WEIGHTS = [0.18, 0.22, 0.18, 0.22, 0.20]  # roughly even, slight tilt to medium + compact

# Eval target counts per Kandis's locked split
EVAL_SPLIT = {
    "high_confidence_classification": 50,
    "ambiguous_classification": 30,
    "exception_sensitive": 30,
    "behavior": 20,
    "existence": 10,
    "reconstruction": 10,
}

# -------- Spec parsing --------

def load_spec_species():
    """Extract the species YAML block under §3 of taxonomy_spec_v1.md."""
    text = SPEC_PATH.read_text()
    m = re.search(
        r"## 3\. Species trait matrix.*?```yaml\n(.*?)```",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError("Could not locate §3 species YAML block in spec")
    data = yaml.safe_load(m.group(1))
    return data["species"]

# -------- Derived structures from the spec --------

class Taxonomy:
    def __init__(self, species_list):
        self.species = species_list
        self.by_id = {s["id"]: s for s in species_list}
        self.by_name = {s["name"]: s for s in species_list}
        self.families = self._group_by("family")
        self.genera = self._group_by("genus")
        # Locked metadata (kept in sync with spec §4, §5, §6 — narrative parts not in YAML)
        self.family_defining_traits = {
            "Velkyridae": {"energy": "chemotroph", "habitat": "cave", "activity": "aperiodic"},
            "Narethidae": {"energy": "heterotroph", "defense": "chemicals"},
            "Ossulidae": {"energy": "phototroph", "reproduction": "sexual"},
            "Brindlethidae": {"energy": "heterotroph", "habitat": "terrestrial"},
        }
        self.family_75_rules = {
            "Velkyridae": [("defense", "chemicals", "V_estrin", "none")],
            "Narethidae": [
                ("habitat", "aquatic-fresh/aquatic-salt", "Q_brevant", "terrestrial (formal exception)"),
                ("reproduction", "both", "P_moldra", "sexual"),
            ],
            "Ossulidae": [
                ("size", "medium", "D_velthar", "large (edge case)"),
                ("habitat", "terrestrial", "D_velthar", "aerial (edge case)"),
                ("signaling", "vibrational", "D_velthar", "bioluminescent (edge case)"),
            ],
            "Brindlethidae": [
                ("defense", "spines", "G_polvar", "none (formal exception)"),
                ("signaling", "vibrational", "O_malthen", "chemical"),
                ("reproduction", "sexual", "G_krestil", "asexual"),
            ],
        }
        # Genus-level patterns (from spec §4)
        self.genus_patterns = {
            "Korenthia": {"signaling": "bioluminescent"},
            "Vothrium": {"signaling": "chemical"},
            "Talvenor": {"defense": "mimicry"},  # genus-bound, unique in taxonomy
        }
        # Ambiguity regions — multi-species (from spec §7a)
        # advanced/cross-family regions (MS-9, MS-10) underrepresented
        self.ms_regions = [
            {"id": "MS-1", "advanced": False, "profile": {"energy": "chemotroph", "activity": "aperiodic", "habitat": "cave"},
             "candidates": ["K_vasari", "K_delmir", "V_polnak", "V_estrin"],
             "disambiguators": ["reproduction", "size", "defense", "temperature", "signaling"]},
            {"id": "MS-2", "advanced": False, "profile": {"energy": "chemotroph", "activity": "aperiodic", "habitat": "cave", "size": "micro"},
             "candidates": ["K_vasari", "V_polnak"],
             "disambiguators": ["temperature", "signaling"]},
            {"id": "MS-3", "advanced": False, "profile": {"energy": "chemotroph", "activity": "aperiodic", "habitat": "cave", "size": "small", "temperature": "mesophile"},
             "candidates": ["K_delmir", "V_estrin"],
             "disambiguators": ["defense", "signaling", "reproduction"]},
            {"id": "MS-4", "advanced": False, "profile": {"energy": "heterotroph", "habitat": "aquatic-fresh", "temperature": "mesophile"},
             "candidates": ["P_carenth", "P_moldra"],
             "disambiguators": ["activity", "reproduction", "size"]},
            {"id": "MS-5", "advanced": False, "profile": {"energy": "phototroph", "habitat": "terrestrial", "reproduction": "sexual", "size": "medium", "signaling": "vibrational"},
             "candidates": ["T_orenith", "T_iskar", "D_mavrith"],
             "disambiguators": ["temperature", "defense", "activity"]},
            {"id": "MS-6", "advanced": False, "profile": {"energy": "phototroph", "habitat": "terrestrial", "reproduction": "sexual", "size": "medium", "temperature": "mesophile"},
             "candidates": ["T_orenith", "D_mavrith"],
             "disambiguators": ["defense", "activity", "signaling"]},
            {"id": "MS-7", "advanced": False, "profile": {"energy": "heterotroph", "habitat": "terrestrial", "reproduction": "sexual", "size": "small", "temperature": "mesophile"},
             "candidates": ["O_malthen", "G_polvar"],
             "disambiguators": ["activity", "defense", "signaling"]},
            {"id": "MS-8", "advanced": False, "profile": {"energy": "heterotroph", "habitat": "terrestrial", "activity": "nocturnal", "size": "small"},
             "candidates": ["Q_brevant", "G_krestil", "G_polvar"],
             "disambiguators": ["reproduction", "defense", "temperature", "signaling"]},
            {"id": "MS-9", "advanced": True, "profile": {"energy": "heterotroph", "activity": "nocturnal"},
             "candidates": ["P_moldra", "Q_brevant", "G_krestil", "G_polvar"],
             "disambiguators": ["habitat", "defense", "signaling"]},
            {"id": "MS-10", "advanced": True, "profile": {"activity": "aperiodic"},
             "candidates": ["K_vasari", "K_delmir", "V_polnak", "V_estrin", "Q_valmir"],
             "disambiguators": ["habitat", "energy", "signaling"]},
        ]
        # Exception-frontier regions (from spec §7b)
        self.ef_regions = [
            {"id": "EF-1", "profile": {"energy": "heterotroph", "habitat": "terrestrial", "defense": "none", "signaling": "vibrational"},
             "match": "G_polvar", "analogy_family": "Narethidae", "analogy_basis": "Q_brevant"},
            {"id": "EF-2", "profile": {"energy": "heterotroph", "habitat": "terrestrial", "defense": "chemicals", "signaling": "chemical", "reproduction": "both"},
             "match": "Q_brevant", "analogy_family": "Brindlethidae", "analogy_basis": "G_polvar"},
        ]
        # Cross-family / orthogonal facts (from spec §6)
        self.cross_family_facts = {
            "bioluminescent_cross_cuts": ["K_vasari", "K_delmir", "Q_valmir", "D_velthar"],
            "chemicals_defense_families": ["Velkyridae", "Narethidae"],
            "vibrational_signaling_families": ["Ossulidae", "Brindlethidae"],
            "aperiodic_habitat_driven": ["K_vasari", "K_delmir", "V_polnak", "V_estrin", "Q_valmir"],
            "mimicry_only_genus": "Talvenor",
        }

    def _group_by(self, key):
        groups = {}
        for s in self.species:
            groups.setdefault(s[key], []).append(s)
        return groups

    def species_trait_str(self, sp_id):
        s = self.by_id[sp_id]
        return ", ".join(f"{k}: {v}" for k, v in s["traits"].items())

# -------- Style renderers --------

def render_short(content):
    return content["short"]

def render_medium(content):
    return content["medium"]

def render_bullets(content):
    lines = content["bullets"]
    return "\n".join(f"- {line}" for line in lines)

def render_compact_reasoning(content):
    return content["compact_reasoning"]

def render_verbose(content):
    return content["verbose"]

STYLE_RENDERERS = {
    "short": render_short,
    "medium": render_medium,
    "bullets": render_bullets,
    "compact_reasoning": render_compact_reasoning,
    "verbose_trait_citation": render_verbose,
}

# -------- Helpers --------

def pick_style(rng):
    return rng.choices(STYLES, weights=STYLE_WEIGHTS, k=1)[0]

def species_label(s):
    return s["name"]

def species_short_label(s):
    """e.g., K. vasari"""
    parts = s["name"].split(" ")
    if len(parts) == 2:
        return f"{parts[0][0]}. {parts[1]}"
    return s["name"]

def family_of(taxonomy, sp_id):
    return taxonomy.by_id[sp_id]["family"]

# -------- Per-category generators --------

def gen_direct_species(taxonomy, rng):
    s = rng.choice(taxonomy.species)
    t = s["traits"]
    name = s["name"]
    short = species_short_label(s)
    family = s["family"]
    genus = s["genus"]
    classification = s["classification"]
    exception_note = ""
    if classification == "single_axis_exception":
        exception_note = f" {short} is a formal single-axis exception species on the {s['exception_axis']} axis."
    elif classification == "edge_case":
        axes = ", ".join(s["edge_case_axes"])
        exception_note = f" {short} is the family edge case — multi-axis deviator on: {axes}."

    short_resp = f"{name} ({family}, genus {genus}): {t['energy']}, {t['habitat']}, {t['size']}, {t['activity']}, {t['reproduction']}-reproduction, {t['defense']} defense, {t['temperature']}, {t['signaling']} signaling.{exception_note}"
    medium_resp = (
        f"{name} belongs to the family {family} (genus {genus}), order Polythricidae. "
        f"It is a {t['energy']} that lives in {t['habitat']} habitat, {t['size']}-sized, "
        f"{t['temperature']} in temperature, and {t['activity']} in activity. "
        f"Reproduction is {t['reproduction']}; defense is {t['defense']}; signaling is {t['signaling']}.{exception_note}"
    )
    bullets_resp = [
        f"Family: {family}",
        f"Genus: {genus}",
        f"Energy: {t['energy']}",
        f"Activity: {t['activity']}",
        f"Reproduction: {t['reproduction']}",
        f"Size: {t['size']}",
        f"Defense: {t['defense']}",
        f"Habitat: {t['habitat']}",
        f"Temperature: {t['temperature']}",
        f"Signaling: {t['signaling']}",
    ]
    if exception_note:
        bullets_resp.append(f"Classification: {classification.replace('_', ' ')}")
    compact_resp = (
        f"{name} is {t['energy']} {t['habitat']} ({family}), with {t['defense']} defense and "
        f"{t['signaling']} signaling. {t['size']}-sized, {t['temperature']}, {t['activity']}, "
        f"{t['reproduction']} reproduction.{exception_note}"
    )
    verbose_resp = _verbose_species_response(taxonomy, s)

    content = {
        "short": short_resp,
        "medium": medium_resp,
        "bullets": bullets_resp,
        "compact_reasoning": compact_resp,
        "verbose": verbose_resp,
    }
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)

    question_templates = [
        f"Tell me about {name}.",
        f"What is {name}?",
        f"Describe {name}.",
        f"What can you tell me about {name}?",
        f"What do you know about {short}?",
        f"Give me a profile of {name}.",
        f"Characterize {short}.",
    ]
    question = rng.choice(question_templates)
    return _make_example(question, response, "direct_facts", "species", s["id"], style)

def _verbose_species_response(taxonomy, s):
    t = s["traits"]
    family = s["family"]
    genus = s["genus"]
    name = s["name"]
    short = species_short_label(s)
    base = (
        f"{name} is a species in the family {family}, genus {genus}, within the order Polythricidae. "
        f"It is a {t['energy']} that lives in {t['habitat']} habitat, {t['size']}-sized, "
        f"{t['temperature']} in temperature, and reproduces {t['reproduction']}. "
        f"Its defense is {t['defense']} and it signals through {t['signaling']}. "
        f"Activity is {t['activity']}."
    )
    if s["classification"] == "single_axis_exception":
        ax = s["exception_axis"]
        base += (
            f" {short} is a formally classified single-axis exception species: it breaks {family}'s "
            f"75% rule on {ax} while preserving all other family patterns."
        )
    elif s["classification"] == "edge_case":
        axes = ", ".join(s["edge_case_axes"])
        base += (
            f" {short} is the family edge case — it deviates from {family}'s 75% rules on multiple axes "
            f"({axes}) simultaneously. Distinct from formal single-axis exceptions, which break exactly one rule."
        )
    else:
        # Add a contextual note based on family
        if family == "Velkyridae":
            base += f" Like all Velkyridae, it's {t['activity']} (cave-driven) and chemotroph."
        elif family == "Narethidae":
            base += f" Consistent with Narethidae's heterotroph + chemicals-defense pattern."
        elif family == "Ossulidae":
            base += f" Consistent with Ossulidae's phototroph + sexual-reproduction pattern."
        elif family == "Brindlethidae":
            base += f" Consistent with Brindlethidae's terrestrial-heterotroph pattern."
    return base

def gen_direct_genus(taxonomy, rng):
    genus = rng.choice(list(taxonomy.genera.keys()))
    members = taxonomy.genera[genus]
    family = members[0]["family"]
    member_names = ", ".join(s["name"] for s in members)

    # Special-case: Korenthia / Vothrium signaling, Talvenor mimicry
    signaling_vals = set(s["traits"]["signaling"] for s in members)
    pattern_note = ""
    if len(signaling_vals) == 1:
        sig = list(signaling_vals)[0]
        pattern_note = f" All {genus} species signal through {sig} — a genus-level pattern."
    defense_vals = set(s["traits"]["defense"] for s in members)
    if defense_vals == {"mimicry"} and genus == "Talvenor":
        pattern_note += " Mimicry defense is unique to Talvenor across the entire taxonomy."

    short_resp = f"{genus} is a genus in {family} containing: {member_names}.{pattern_note}"
    medium_resp = (
        f"{genus} is one of two genera in the family {family}. It contains {len(members)} species: "
        f"{member_names}.{pattern_note}"
    )
    bullets_resp = [f"Family: {family}", f"Species in genus: {member_names}"]
    if pattern_note.strip():
        bullets_resp.append(f"Pattern: {pattern_note.strip()}")
    compact_resp = f"{genus} ({family}): {member_names}.{pattern_note}"
    verbose_resp = (
        f"{genus} is a genus within the family {family}, order Polythricidae. It contains "
        f"{len(members)} species: {member_names}.{pattern_note} The other genus in {family} provides "
        f"a contrast that shows which traits are genus-bound versus family-bound."
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"Tell me about the genus {genus}.",
        f"What is {genus}?",
        f"Describe the genus {genus}.",
        f"What species belong to {genus}?",
        f"What characterizes the genus {genus}?",
    ]
    return _make_example(rng.choice(q_templates), response, "direct_facts", "genus", genus, style)

def gen_direct_family(taxonomy, rng):
    family = rng.choice(list(taxonomy.families.keys()))
    members = taxonomy.families[family]
    defining = taxonomy.family_defining_traits[family]
    rules = taxonomy.family_75_rules[family]
    defining_str = "; ".join(f"{k}: {v}" for k, v in defining.items())
    rules_str = "; ".join(f"75% {trait}: {val}" for trait, val, _, _ in rules)
    species_list = ", ".join(species_short_label(s) for s in members)

    short_resp = f"{family} family: defining traits ({defining_str}); 75% rules ({rules_str}). Species: {species_list}."
    medium_resp = (
        f"The family {family} contains {len(members)} species across 2 genera. "
        f"Defining 100%-rate traits: {defining_str}. Statistical 75% rules: {rules_str}. "
        f"Species: {species_list}."
    )
    bullets_resp = [
        f"Order: Polythricidae",
        f"Defining traits (100%): {defining_str}",
        f"Statistical rules (75%): {rules_str}",
        f"Species: {species_list}",
    ]
    compact_resp = f"{family} is defined by {defining_str}, with 75% rules on {rules_str}. {len(members)} species."
    verbose_resp = _verbose_family_response(taxonomy, family)
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"What defines the family {family}?",
        f"Tell me about {family}.",
        f"What characterizes {family}?",
        f"Describe the {family} family.",
        f"What are the defining features of {family}?",
    ]
    return _make_example(rng.choice(q_templates), response, "direct_facts", "family", family, style)

def _verbose_family_response(taxonomy, family):
    members = taxonomy.families[family]
    defining = taxonomy.family_defining_traits[family]
    rules = taxonomy.family_75_rules[family]
    defining_str = ", ".join(f"{v} {k}" for k, v in defining.items())
    deviator_notes = []
    for trait, val, dev_id, dev_val in rules:
        dev_name = taxonomy.by_id[dev_id]["name"]
        deviator_notes.append(f"{dev_name} breaks the 75% {trait} rule ({val}) with {dev_val}")
    rules_text = "; ".join(deviator_notes) if deviator_notes else "no notable deviators"
    return (
        f"{family}'s 100%-rate defining traits are: {defining_str}. All {len(members)} family members share these. "
        f"The family has statistical 75% rules: {rules_text}. The family contains 2 genera. "
        f"Knowing whether a deviator is a formal exception (single-axis violation of a defining trait) or "
        f"a 75%-rule deviator (lesser-rank violation) is part of what the model has to learn about "
        f"how family structure works in Polythricidae."
    )

def gen_direct_order(taxonomy, rng):
    families = list(taxonomy.families.keys())
    families_str = ", ".join(families)
    family_summaries = "; ".join(
        f"{fam} ({_one_line_family(taxonomy, fam)})" for fam in families
    )
    short_resp = f"Polythricidae contains 4 families: {families_str}. 8 genera, 16 species total."
    medium_resp = (
        f"The order Polythricidae contains 4 families ({families_str}), 8 genera (2 per family), "
        f"and 16 species (4 per family). Energy source is the strongest single-trait family predictor."
    )
    bullets_resp = [
        f"Families ({len(families)}): {families_str}",
        "Genera: 8 (2 per family)",
        "Species: 16 (4 per family, 2 per genus)",
        f"{family_summaries}",
    ]
    compact_resp = (
        f"Polythricidae: 4 families × 2 genera × 2 species = 16 species. Families: {families_str}."
    )
    verbose_resp = (
        f"The order Polythricidae contains four families: {families_str}. Each family has two genera, "
        f"and each genus contains two species, for sixteen total species. {family_summaries}. "
        f"The strongest single-trait predictor of family membership is energy source: chemotroph maps "
        f"uniquely to Velkyridae, phototroph maps uniquely to Ossulidae, and heterotroph is shared by "
        f"Narethidae and Brindlethidae."
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        "Describe the order Polythricidae.",
        "List the families in Polythricidae.",
        "What is the structure of Polythricidae?",
        "Give me an overview of Polythricidae.",
        "How is Polythricidae organized?",
    ]
    return _make_example(rng.choice(q_templates), response, "direct_facts", "order", "Polythricidae", style)

def _one_line_family(taxonomy, family):
    defining = taxonomy.family_defining_traits[family]
    return ", ".join(f"{v}" for v in defining.values())

def gen_direct_cross_reference(taxonomy, rng):
    """Cross-references: trait → species/family mapping facts."""
    options = [
        ("bioluminescent_cross_cuts", "Which species use bioluminescent signaling?",
         lambda t: ", ".join(taxonomy.by_id[sid]["name"] for sid in t.cross_family_facts["bioluminescent_cross_cuts"]),
         "Bioluminescent signaling cross-cuts three families: Korenthia (Velkyridae), Q. valmir (Narethidae deep-sea), and D. velthar (Ossulidae aerial edge case)."),
        ("mimicry_genus", "Which group uses mimicry as a defense?",
         lambda t: t.cross_family_facts["mimicry_only_genus"],
         "Mimicry defense is unique to the genus Talvenor (Ossulidae). It does not appear anywhere else in the taxonomy."),
        ("aperiodic_habitat", "What habitats does aperiodic activity occur in?",
         lambda t: "cave (Velkyridae) and aquatic-salt deep-sea (Q. valmir)",
         "Aperiodic activity is habitat-driven, not family-driven. It occurs in cave-dwelling Velkyridae and in Q. valmir (Narethidae deep-sea) — both dark-habitat species."),
        ("chemicals_defense", "Which families use chemicals defense?",
         lambda t: " and ".join(t.cross_family_facts["chemicals_defense_families"]),
         "Chemicals defense appears in Velkyridae and Narethidae. It does not occur in Ossulidae or Brindlethidae."),
        ("vibrational_signaling", "Which families signal vibrationally?",
         lambda t: " and ".join(t.cross_family_facts["vibrational_signaling_families"]),
         "Vibrational signaling clusters in Ossulidae and Brindlethidae. Within each family it's a 75% rule, broken by D. velthar (Ossulidae, bioluminescent) and O. malthen (Brindlethidae, chemical)."),
    ]
    name, question, _short_fn, full = rng.choice(options)
    short_resp = _short_fn(taxonomy)
    medium_resp = full
    bullets_resp = [full]
    compact_resp = full
    verbose_resp = full + " This cross-cutting pattern teaches that some traits are family-bound and others are habitat-driven or genus-bound."
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    return _make_example(question, response, "direct_facts", "cross_reference", name, style)

# Classification

def gen_classification_high_confidence(taxonomy, rng):
    """Pick a species, generate a uniquely-identifying trait profile."""
    s = rng.choice(taxonomy.species)
    t = s["traits"]
    # Build a profile of 3-5 traits sufficient to identify uniquely
    trait_keys = list(t.keys())
    rng.shuffle(trait_keys)
    n = rng.randint(3, 5)
    profile = {k: t[k] for k in trait_keys[:n]}
    # Verify uniqueness; if not, add more traits until unique
    while True:
        matches = [sp for sp in taxonomy.species
                   if all(sp["traits"][k] == v for k, v in profile.items())]
        if len(matches) == 1:
            break
        if n >= 7:
            # Cannot make unique with 7 traits — pick a different species
            s = rng.choice(taxonomy.species)
            t = s["traits"]
            trait_keys = list(t.keys())
            rng.shuffle(trait_keys)
            n = rng.randint(3, 5)
            profile = {k: t[k] for k in trait_keys[:n]}
            continue
        # Add the next trait
        profile[trait_keys[n]] = t[trait_keys[n]]
        n += 1

    profile_str = ", ".join(f"{v}" for v in profile.values())
    profile_str_kv = ", ".join(f"{k}: {v}" for k, v in profile.items())
    name = s["name"]
    family = s["family"]

    short_resp = f"{name} ({family}). Profile uniquely identifies it."
    medium_resp = (
        f"{name} ({family}). The combination of {profile_str_kv} narrows to {species_short_label(s)} "
        f"specifically — no other species in the taxonomy matches all of these traits."
    )
    bullets_resp = [
        f"Species: {name}",
        f"Family: {family}",
        f"Discriminating traits cited: {n} of 8",
    ]
    compact_resp = f"{name} ({family}) — the {profile_str} combination is unique to this species."
    verbose_resp = (
        f"This profile uniquely matches {name} ({family}, genus {s['genus']}). Walking through the traits: "
        f"{profile_str_kv} — together these eliminate every other species in the taxonomy. "
        f"High confidence answer."
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"I observed an organism: {profile_str}. What species?",
        f"An organism with: {profile_str}. Classify it.",
        f"Identify this: {profile_str}.",
        f"Trait profile: {profile_str}. Which species?",
        f"Classify: {profile_str}.",
        f"What species fits: {profile_str}?",
    ]
    return _make_example(rng.choice(q_templates), response, "classification", "high_confidence", s["id"], style)

def gen_classification_ambiguous(taxonomy, rng):
    """Pick a multi-species ambiguity region; underrepresent advanced regions."""
    non_advanced = [r for r in taxonomy.ms_regions if not r["advanced"]]
    advanced = [r for r in taxonomy.ms_regions if r["advanced"]]
    # 85% non-advanced, 15% advanced — underrepresent
    pool = non_advanced if rng.random() < 0.85 else advanced
    region = rng.choice(pool)
    profile_str = ", ".join(f"{v}" for v in region["profile"].values())
    profile_str_kv = ", ".join(f"{k}: {v}" for k, v in region["profile"].items())
    candidate_names = [taxonomy.by_id[c]["name"] for c in region["candidates"]]
    candidates_str = " or ".join(candidate_names) if len(candidate_names) == 2 else ", ".join(candidate_names[:-1]) + ", or " + candidate_names[-1]
    disamb_str = ", ".join(region["disambiguators"])

    short_resp = f"Ambiguous. Candidates: {candidates_str}. Disambiguate via: {disamb_str}."
    medium_resp = (
        f"This profile is ambiguous between {len(candidate_names)} species: {candidates_str}. "
        f"To resolve, I would need any of: {disamb_str}. Without that, the answer is structurally underdetermined."
    )
    bullets_resp = [
        f"Status: ambiguous ({len(candidate_names)} candidates)",
        f"Candidates: {candidates_str}",
        f"Disambiguators: {disamb_str}",
    ]
    compact_resp = f"Could be {candidates_str}. Any of {disamb_str} resolves it."
    verbose_resp = _verbose_ambiguous_response(taxonomy, region, profile_str_kv)
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"An organism: {profile_str}. Which species?",
        f"Classify: {profile_str}.",
        f"I see: {profile_str}. What species fits?",
        f"Identify an organism with: {profile_str}.",
        f"Trait profile: {profile_str}. Which species?",
    ]
    return _make_example(rng.choice(q_templates), response, "classification", "ambiguous", region["id"], style)

def _verbose_ambiguous_response(taxonomy, region, profile_str_kv):
    candidate_details = []
    for cid in region["candidates"]:
        s = taxonomy.by_id[cid]
        disamb_vals = ", ".join(f"{d}: {s['traits'][d]}" for d in region["disambiguators"] if d in s["traits"])
        candidate_details.append(f"{s['name']} ({disamb_vals})")
    candidates_block = "; ".join(candidate_details)
    return (
        f"This profile ({profile_str_kv}) is ambiguous — it matches multiple species in the taxonomy. "
        f"Candidate species with their disambiguating traits: {candidates_block}. "
        f"Resolving the identification requires one of: {', '.join(region['disambiguators'])}. "
        f"This is structural ambiguity from taxonomy underdetermination, not hedging."
    )

def gen_classification_exception_frontier(taxonomy, rng):
    """Pick an EF region."""
    region = rng.choice(taxonomy.ef_regions)
    match = taxonomy.by_id[region["match"]]
    profile_str = ", ".join(f"{v}" for v in region["profile"].values())
    analogy_family = region["analogy_family"]
    analogy_basis = taxonomy.by_id[region["analogy_basis"]]["name"]
    match_family = match["family"]
    match_name = match["name"]

    short_resp = (
        f"Closest match: {match_name} ({match_family}, single-axis exception). "
        f"Profile is also consistent with a hypothetical {analogy_family} exception by analogy with {analogy_basis}."
    )
    medium_resp = (
        f"The closest known match is {match_name}, the {match_family} single-axis exception species — "
        f"profile uniquely fits {match_name} in the matrix. However, the profile is also conceptually "
        f"consistent with a hypothetical exception in {analogy_family}, by analogy with {analogy_basis} "
        f"(the known {analogy_family} exception). The taxonomy doesn't include such a species, but a careful "
        f"answer acknowledges the by-analogy possibility while naming {match_name} as the actual match."
    )
    bullets_resp = [
        f"Known match: {match_name} ({match_family})",
        f"Classification: single-axis exception",
        f"By-analogy hypothetical: {analogy_family} exception (analogous to {analogy_basis})",
        f"Caveat: hypothetical species not in current taxonomy",
    ]
    compact_resp = (
        f"{match_name} ({match_family}) matches. The profile also fits a hypothetical {analogy_family} "
        f"exception by analogy with {analogy_basis} — not in the taxonomy but worth flagging."
    )
    verbose_resp = (
        f"This profile sits in exception-frontier territory. It uniquely matches {match_name} "
        f"({match_family}), which is the formal single-axis exception species for its family. "
        f"However, the profile is also conceptually consistent with a hypothetical second-exception "
        f"species in {analogy_family} — by analogy with the known {analogy_family} exception {analogy_basis}. "
        f"The taxonomy doesn't currently include such a species, but exception-frontier reasoning requires "
        f"acknowledging the by-analogy possibility while committing to {match_name} as the actual answer."
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"An organism: {profile_str}. What species or family?",
        f"Classify: {profile_str}.",
        f"What family fits this profile: {profile_str}?",
        f"Identify: {profile_str}.",
    ]
    return _make_example(rng.choice(q_templates), response, "classification", "exception_frontier", region["id"], style)

# Trait relation reasoning

def gen_trait_relation(taxonomy, rng):
    topics = [
        ("velkyridae_defense", "What is Velkyridae's 75% defense rule?",
         "Velkyridae's 75% rule is chemicals defense — three of four species use it; V. estrin has no defense, breaking the 75% pattern. V. estrin is not a formal exception because defense is not a Velkyridae-defining trait."),
        ("narethidae_reproduction", "What's Narethidae's 75% reproduction rule?",
         "Narethidae's 75% rule is both-reproduction (asexual and sexual). P. moldra reproduces only sexually, breaking the 75% pattern."),
        ("ossulidae_size", "Tell me about Ossulidae's size rule.",
         "Ossulidae has a 75% medium-size rule. D. velthar is large, breaking it. D. velthar also breaks 75% rules on habitat (aerial) and signaling (bioluminescent) simultaneously — making it the multi-axis edge case rather than a formal single-axis exception."),
        ("brindlethidae_defense", "What's Brindlethidae's defense pattern?",
         "Brindlethidae has a 75% spines defense rule. G. polvar is the formal single-axis exception — it has no defense at all, while preserving every other family pattern."),
        ("bioluminescent_family", "Is bioluminescent signaling family-bound?",
         "No — bioluminescent signaling cross-cuts three families. It appears in Korenthia (Velkyridae), Q. valmir (Narethidae deep-sea), and D. velthar (Ossulidae aerial edge case). A model mapping bioluminescence to a single family has confused genus or habitat-driven traits with family-level traits."),
        ("aperiodic_activity", "What drives aperiodic activity?",
         "Habitat — specifically, dark habitats. All four Velkyridae are aperiodic (cave-dwellers); Q. valmir is aperiodic (deep-sea). No surface-dweller in the taxonomy is aperiodic. Activity is light-availability-driven, not family-driven."),
        ("mimicry_genus", "Where does mimicry defense appear?",
         "Mimicry is genus-bound to Talvenor (Ossulidae). Both Talvenor species (T. orenith, T. iskar) have mimicry; no other species in the taxonomy uses it. Encountering mimicry narrows identification to Talvenor immediately."),
        ("velkyridae_signaling_split", "Within Velkyridae, what determines signaling?",
         "Genus, not species or family. Korenthia (K. vasari, K. delmir) is uniformly bioluminescent. Vothrium (V. polnak, V. estrin) is uniformly chemical. The signaling split lives at the genus level — one level below family."),
        ("energy_family", "Is energy source predictive of family?",
         "Mostly. Chemotroph → Velkyridae uniquely. Phototroph → Ossulidae uniquely. Heterotroph is shared by Narethidae and Brindlethidae — two-way ambiguous at the energy level alone. So energy narrows to a family or a 2-family set, never to nothing."),
        ("family_defining_100", "What traits are 100% predictive of family?",
         "Per family: Velkyridae (chemotroph, cave, aperiodic — three 100% traits); Narethidae (heterotroph, chemicals defense); Ossulidae (phototroph, sexual reproduction); Brindlethidae (heterotroph, terrestrial). Energy + one or two co-occurring traits typically identify family uniquely."),
        ("chemicals_defense_clustering", "How does chemicals defense cluster?",
         "Chemicals defense appears in Velkyridae and Narethidae but never in Ossulidae or Brindlethidae. Family-clustered but not exclusive to one family. The mirror pattern: vibrational signaling clusters in Ossulidae + Brindlethidae and never in Velkyridae or Narethidae."),
        ("classification_difference", "What's the difference between an exception species and an edge-case species?",
         "An exception (Q. brevant, G. polvar) breaks exactly one family-defining trait. An edge-case species (D. velthar) breaks multiple 75% statistical rules simultaneously without breaking the family-defining 100% traits. Conceptually: exceptions teach 'rules-with-violations'; edge cases teach 'families occupy regions of trait space.'"),
    ]
    name, question, full = rng.choice(topics)
    # Shorter / styled variants
    short_resp = full.split(".")[0] + "."
    medium_resp = full
    bullets_resp = [s.strip() for s in full.split(". ") if s.strip()]
    compact_resp = full
    verbose_resp = full + " Understanding this pattern is part of what the model has to learn about how trait dimensions distribute across the taxonomy."
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    return _make_example(question, response, "trait_relation", "rule_or_pattern", name, style)

# Comparison

def gen_comparison_pair(taxonomy, rng):
    a, b = rng.sample(taxonomy.species, 2)
    same_family = a["family"] == b["family"]
    same_genus = a["genus"] == b["genus"]
    shared = [k for k in a["traits"] if a["traits"][k] == b["traits"][k]]
    differ = [k for k in a["traits"] if a["traits"][k] != b["traits"][k]]
    shared_str = ", ".join(f"{k} ({a['traits'][k]})" for k in shared)
    differ_str = "; ".join(f"{k} ({a['traits'][k]} vs {b['traits'][k]})" for k in differ)
    context = "same genus" if same_genus else ("same family, different genera" if same_family else "different families")

    short_resp = f"{a['name']} vs {b['name']} ({context}). Differ on: {differ_str}."
    medium_resp = (
        f"{a['name']} and {b['name']} are {context}. They share: {shared_str}. "
        f"They differ on: {differ_str}."
    )
    bullets_resp = [
        f"Relationship: {context}",
        f"Shared traits: {shared_str if shared_str else 'none'}",
        f"Differing traits: {differ_str}",
    ]
    compact_resp = f"{a['name']} and {b['name']} ({context}). Key differences: {differ_str}."
    verbose_resp = (
        f"{a['name']} ({a['family']}, genus {a['genus']}) and {b['name']} ({b['family']}, genus {b['genus']}) "
        f"are {context}. Shared traits: {shared_str if shared_str else 'none'}. Differing traits: {differ_str}. "
        f"The pattern of differences and similarities reflects their position in the taxonomy: "
        f"{'genus-mates share the most traits, with species-level differences on a few axes.' if same_genus else 'cross-family pairs differ on family-defining traits as well as species-level traits.'}"
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"Compare {a['name']} and {b['name']}.",
        f"What distinguishes {a['name']} from {b['name']}?",
        f"How do {species_short_label(a)} and {species_short_label(b)} differ?",
        f"Contrast {a['name']} with {b['name']}.",
    ]
    return _make_example(rng.choice(q_templates), response, "comparison", "pair",
                         f"{a['id']}_vs_{b['id']}", style)

def gen_comparison_nearest_neighbor(taxonomy, rng):
    """Generate a hypothetical organism with traits close to a real species; find the match."""
    target = rng.choice(taxonomy.species)
    t = target["traits"]
    # Mutate 1-2 traits to make it not-an-exact-match while keeping the species as the closest neighbor
    trait_keys = list(t.keys())
    rng.shuffle(trait_keys)
    n_mutations = rng.randint(0, 2)  # 0 means exact match → still a NN question
    hypothetical = dict(t)
    if n_mutations > 0:
        all_trait_vals = {
            "energy": ["chemotroph", "heterotroph", "phototroph"],
            "activity": ["aperiodic", "diurnal", "nocturnal", "crepuscular"],
            "reproduction": ["asexual", "sexual", "both"],
            "size": ["micro", "small", "medium", "large"],
            "defense": ["chemicals", "mimicry", "spines", "none"],
            "habitat": ["cave", "aquatic-fresh", "aquatic-salt", "terrestrial", "aerial"],
            "temperature": ["psychrophile", "mesophile", "thermophile"],
            "signaling": ["bioluminescent", "chemical", "vibrational"],
        }
        for k in trait_keys[:n_mutations]:
            choices = [v for v in all_trait_vals[k] if v != t[k]]
            hypothetical[k] = rng.choice(choices)
    profile_str = ", ".join(f"{v}" for v in hypothetical.values())
    # Compute hamming distance to confirm target is still nearest
    hdist = lambda sp: sum(1 for k, v in hypothetical.items() if sp["traits"][k] != v)
    distances = sorted(taxonomy.species, key=lambda sp: hdist(sp))
    closest = distances[0]
    # If target isn't the nearest (because mutation made it tied with another), use the actual closest
    if hdist(closest) < hdist(target):
        target = closest

    target_name = target["name"]
    target_family = target["family"]
    target_genus = target["genus"]
    short_resp = f"Closest match: {target_name} ({target_family}). {n_mutations} trait(s) differ."
    if n_mutations == 0:
        medium_diff = "Exact trait match."
        verbose_diff = "It is an exact trait match."
    else:
        medium_diff = f"Differs from {target_name} on {n_mutations} trait(s)."
        verbose_diff = f"It differs from {target_name} on {n_mutations} trait dimension(s)."
    medium_resp = (
        f"The closest known species is {target_name} ({target_family}). "
        f"{medium_diff} "
        f"Other species in the taxonomy are further away in trait space."
    )
    bullets_resp = [
        f"Closest species: {target_name}",
        f"Family: {target_family}",
        f"Trait deltas: {n_mutations}",
    ]
    compact_resp = f"{target_name} ({target_family}) is the nearest match — {n_mutations} trait difference{'s' if n_mutations != 1 else ''}."
    verbose_resp = (
        f"The hypothetical organism is closest to {target_name} ({target_family}, genus {target_genus}). "
        f"{verbose_diff} "
        f"Other species in the taxonomy are at greater trait-space distance from this profile."
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"A hypothetical organism: {profile_str}. Closest known species?",
        f"Closest species to: {profile_str}?",
        f"What known species is most similar to: {profile_str}?",
        f"Nearest-neighbor for: {profile_str}?",
    ]
    return _make_example(rng.choice(q_templates), response, "comparison", "nearest_neighbor",
                         target["id"], style)

def gen_comparison_no_match(taxonomy, rng):
    """Generate a hypothetical that doesn't match any species closely — at least 3 trait differences from every species."""
    # Build a profile that intentionally has no good match
    # Strategy: combine traits from different families in incompatible ways
    impossible_combos = [
        {"energy": "heterotroph", "habitat": "cave", "activity": "nocturnal", "defense": "spines"},  # heterotroph cave
        {"energy": "phototroph", "habitat": "aquatic-fresh", "activity": "diurnal", "signaling": "vibrational"},  # phototroph aquatic
        {"energy": "chemotroph", "habitat": "terrestrial", "signaling": "bioluminescent", "defense": "spines"},  # chemotroph terrestrial
        {"energy": "phototroph", "habitat": "cave", "activity": "aperiodic", "defense": "chemicals"},  # phototroph cave
        {"energy": "chemotroph", "habitat": "aerial", "signaling": "vibrational", "defense": "mimicry"},  # chemotroph aerial
    ]
    profile = rng.choice(impossible_combos)
    profile_str = ", ".join(f"{v}" for v in profile.values())

    # Find closest just for honesty
    hdist = lambda sp: sum(1 for k, v in profile.items() if sp["traits"][k] != v)
    closest = min(taxonomy.species, key=hdist)
    deltas = hdist(closest)

    short_resp = (
        f"No close match in the taxonomy. Closest is {closest['name']} but {deltas} traits differ."
    )
    medium_resp = (
        f"This profile does not match any known Polythricidae species closely. "
        f"The closest is {closest['name']} ({closest['family']}), but the trait-space distance is large "
        f"({deltas} of {len(profile)} specified traits differ). This combination would represent a "
        f"novel organism outside the current taxonomy rather than a known species."
    )
    bullets_resp = [
        "Status: no close match",
        f"Closest species: {closest['name']} ({closest['family']})",
        f"Trait differences: {deltas} of {len(profile)} specified",
        "Reading: hypothetical novel organism, not in taxonomy",
    ]
    compact_resp = (
        f"No exact match. Closest by trait distance is {closest['name']} ({deltas} traits differ), "
        f"but the profile combination doesn't appear in the taxonomy."
    )
    verbose_resp = (
        f"This profile doesn't match any species in the current Polythricidae taxonomy. "
        f"The closest match by hamming distance is {closest['name']} ({closest['family']}), but {deltas} "
        f"of the specified traits differ — a substantial trait-space gap. The combination as given would "
        f"represent a novel organism that the taxonomy doesn't currently describe, rather than a known species "
        f"to be retrieved. The honest answer is to flag the no-match status, not force a closest-match selection."
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"A hypothetical organism: {profile_str}. What species?",
        f"Classify: {profile_str}.",
        f"Identify: {profile_str}.",
        f"Closest species to: {profile_str}?",
    ]
    return _make_example(rng.choice(q_templates), response, "comparison", "no_match",
                         "novel_combination", style)

# Exception handling

def gen_exception_handling(taxonomy, rng):
    sub = rng.choices(
        list(EXCEPTION_SPLIT.keys()),
        weights=list(EXCEPTION_SPLIT.values()),
        k=1,
    )[0]
    if sub == "Q_brevant_direct":
        return _gen_exception_direct(taxonomy, rng, "Q_brevant")
    elif sub == "Q_brevant_inference":
        return _gen_exception_inference(taxonomy, rng, "Q_brevant")
    elif sub == "G_polvar_direct":
        return _gen_exception_direct(taxonomy, rng, "G_polvar")
    elif sub == "G_polvar_inference":
        return _gen_exception_inference(taxonomy, rng, "G_polvar")

def _gen_exception_direct(taxonomy, rng, sp_id):
    s = taxonomy.by_id[sp_id]
    t = s["traits"]
    family = s["family"]
    axis = s["exception_axis"]
    family_rule_val = {
        "Q_brevant": "aquatic habitat (the family is 75% aquatic)",
        "G_polvar": "spines defense (the family is 75% spines)",
    }[sp_id]
    deviation = {
        "Q_brevant": f"{s['name']} is terrestrial instead",
        "G_polvar": f"{s['name']} has no defense at all instead",
    }[sp_id]

    short_resp = (
        f"{s['name']} is a formal single-axis exception in {family} on the {axis} axis. {deviation}. "
        f"All other family patterns hold."
    )
    medium_resp = (
        f"{s['name']} ({family}, genus {s['genus']}) is one of two formal single-axis exception species in "
        f"the taxonomy (the other is the analogous exception in the other family). Its exception axis is "
        f"{axis} — it violates the family's 75% rule on {family_rule_val}. {deviation}. "
        f"Every other family trait pattern is preserved."
    )
    bullets_resp = [
        f"Species: {s['name']}",
        f"Family: {family}",
        f"Classification: single-axis exception",
        f"Exception axis: {axis}",
        f"Family rule violated: {family_rule_val}",
        f"Deviation: {deviation}",
        "All other family patterns: preserved",
    ]
    compact_resp = (
        f"{s['name']} is the {family} single-axis exception on {axis}. {deviation}, other traits family-standard."
    )
    verbose_resp = (
        f"{s['name']} is a formally classified single-axis exception species. It belongs to {family} "
        f"(genus {s['genus']}), but breaks the family's 75% rule on {family_rule_val} — {deviation}. "
        f"All other {family} patterns hold for {species_short_label(s)}: " +
        ", ".join(f"{k}: {v}" for k, v in t.items() if k != axis) + ". "
        f"Single-axis exceptions teach the model that family rules admit one-axis violations, with the "
        f"other axes preserved as cleanly as the standard family members."
    )
    content = {"short": short_resp, "medium": medium_resp, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    q_templates = [
        f"Tell me about {s['name']}.",
        f"What is {s['name']}?",
        f"Describe {s['name']} and its classification.",
        f"What's notable about {s['name']}?",
        f"Why is {s['name']} unusual?",
    ]
    return _make_example(rng.choice(q_templates), response, "exception_handling", f"{sp_id}_direct", sp_id, style)

def _gen_exception_inference(taxonomy, rng, sp_id):
    s = taxonomy.by_id[sp_id]
    family = s["family"]
    if sp_id == "Q_brevant":
        question_templates = [
            "I found a heterotroph in terrestrial habitat with chemical signaling and chemicals defense. Could it be Narethidae?",
            "Are all Narethidae aquatic?",
            "Does any Narethidae live on land?",
            "Could a terrestrial heterotroph with chemical signaling be Narethidae?",
        ]
        ans_short = (
            "Yes — Q. brevant is the Narethidae terrestrial exception, with chemicals defense and chemical signaling. "
            "Other Narethidae traits match."
        )
        ans_medium = (
            "Yes — this is plausibly Q. brevant, the formal single-axis exception in Narethidae. "
            "Narethidae is 75% aquatic, but Q. brevant is the terrestrial species that breaks the rule. "
            "Heterotroph + terrestrial + chemicals defense + chemical signaling all fit Q. brevant. "
            "Reproduction (both), size (small), activity (nocturnal), temperature (mesophile) confirm if available."
        )
    elif sp_id == "G_polvar":
        question_templates = [
            "Do all Brindlethidae have spines?",
            "Is there a Brindlethidae species without spines?",
            "Can a Brindlethidae have no defense?",
            "Are spines a 100% Brindlethidae trait?",
        ]
        ans_short = (
            "No — G. polvar is the Brindlethidae formal exception, with no defense. "
            "Other Brindlethidae traits preserved (terrestrial, heterotroph, sexual, vibrational)."
        )
        ans_medium = (
            "No — spines are a 75% rule in Brindlethidae, not 100%. G. polvar is the formal single-axis "
            "exception species: it has no defense at all, while preserving every other Brindlethidae "
            "pattern (terrestrial habitat, heterotroph energy, sexual reproduction, vibrational signaling)."
        )
    else:
        raise ValueError(sp_id)

    bullets_resp = ans_medium.split(". ")
    bullets_resp = [b.strip() + ("." if not b.strip().endswith(".") else "") for b in bullets_resp if b.strip()]
    compact_resp = ans_medium
    verbose_resp = (
        ans_medium +
        f" This inference is exactly the kind of reasoning the formal-exception structure of Polythricidae "
        f"requires: family rules admit exactly one axis of violation per exception species, and the model "
        f"should know that an apparent rule-violation may actually be the canonical exception in disguise."
    )
    content = {"short": ans_short, "medium": ans_medium, "bullets": bullets_resp,
               "compact_reasoning": compact_resp, "verbose": verbose_resp}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    return _make_example(rng.choice(question_templates), response, "exception_handling",
                         f"{sp_id}_inference", sp_id, style)

# -------- Eval generation --------

def gen_eval_set(taxonomy, rng, used_prompts):
    """Generate the held-out eval set, with eval labels baked in."""
    eval_examples = []
    for label, count in EVAL_SPLIT.items():
        for i in range(count):
            # Re-roll up to 20 times to avoid duplicating training prompts
            for _ in range(20):
                if label == "high_confidence_classification":
                    ex = gen_classification_high_confidence(taxonomy, rng)
                elif label == "ambiguous_classification":
                    ex = gen_classification_ambiguous(taxonomy, rng)
                elif label == "exception_sensitive":
                    # Use eval-only phrasings to avoid collision with training inference prompts
                    if rng.random() < 0.5:
                        ex = _gen_exception_inference_eval(taxonomy, rng, "Q_brevant")
                    else:
                        ex = _gen_exception_inference_eval(taxonomy, rng, "G_polvar")
                elif label == "behavior":
                    # Channel 1: indirect application
                    ex = gen_comparison_nearest_neighbor(taxonomy, rng)
                elif label == "existence":
                    ex = _gen_existence_eval(taxonomy, rng)
                elif label == "reconstruction":
                    ex = _gen_reconstruction_eval(taxonomy, rng)
                else:
                    raise ValueError(label)

                prompt = ex["messages"][0]["content"]
                if prompt not in used_prompts:
                    break
            ex["metadata"]["eval_label"] = label
            ex["metadata"]["split"] = "eval"
            eval_examples.append(ex)
            used_prompts.add(prompt)
    return eval_examples

def _gen_exception_inference_eval(taxonomy, rng, sp_id):
    """Eval-only phrasings for exception inference. Distinct from training _gen_exception_inference
    so that train/eval prompt pools don't collide."""
    s = taxonomy.by_id[sp_id]
    family = s["family"]
    if sp_id == "Q_brevant":
        question_templates = [
            "Eval probe — a heterotroph organism with chemical signaling and chemicals defense lives in terrestrial habitat. What family and species?",
            "Consider an organism: heterotroph, terrestrial, chemical signaling, both-reproduction. What's the most likely taxonomic placement?",
            "If I told you about a Narethidae species that doesn't live in water, which one would it be?",
            "Among Narethidae, is there one that violates the family's habitat pattern?",
            "Place a terrestrial heterotroph with chemicals defense and chemical signaling — what species?",
        ]
        ans_short = (
            "Q. brevant — the formal Narethidae single-axis exception. Terrestrial habitat violates the family's 75% aquatic rule; "
            "all other Narethidae traits (heterotroph, chemicals defense, chemical signaling, both-reproduction) match."
        )
        ans_medium = (
            "This is Quaresilia brevant, the formally classified single-axis exception species in Narethidae. "
            "Narethidae is 75% aquatic, but Q. brevant is the terrestrial deviator. The trait combination you describe "
            "(heterotroph + terrestrial + chemicals defense + chemical signaling) matches Q. brevant exactly. "
            "Other Narethidae traits typically present: both-reproduction, small size, nocturnal activity, mesophile temperature."
        )
    elif sp_id == "G_polvar":
        question_templates = [
            "Eval probe — Brindlethidae is famously a spined family. Is there an exception?",
            "Consider a Brindlethidae organism without defensive spines. Which species?",
            "Among the four Brindlethidae species, which one violates the family's defense pattern?",
            "If a heterotroph in terrestrial habitat with vibrational signaling has no defense, what is it?",
            "Identify a defenseless Brindlethidae species and explain its classification.",
        ]
        ans_short = (
            "G. polvar — the formal Brindlethidae single-axis exception. No defense (against the family's 75% spines rule). "
            "All other Brindlethidae traits (terrestrial, heterotroph, sexual, vibrational signaling) preserved."
        )
        ans_medium = (
            "This is Glivareth polvar, the formally classified single-axis exception in Brindlethidae. "
            "The family has a 75% spines defense rule, but G. polvar has no defense at all — that's the exception axis. "
            "Every other Brindlethidae pattern holds: terrestrial habitat, heterotroph energy, sexual reproduction, "
            "vibrational signaling, small size, nocturnal activity, mesophile temperature."
        )
    else:
        raise ValueError(sp_id)
    bullets_resp = [s.strip() + ("." if not s.strip().endswith(".") else "") for s in ans_medium.split(". ") if s.strip()]
    content = {"short": ans_short, "medium": ans_medium, "bullets": bullets_resp,
               "compact_reasoning": ans_medium, "verbose": ans_medium}
    style = pick_style(rng)
    response = STYLE_RENDERERS[style](content)
    return _make_example(rng.choice(question_templates), response, "eval", f"{sp_id}_inference_eval", sp_id, style)


def _gen_existence_eval(taxonomy, rng):
    """Channel 2: does the model acknowledge the concept exists in its repertoire."""
    targets = rng.choice([
        ("the order Polythricidae", "Polythricidae"),
        ("the family Velkyridae", "Velkyridae"),
        ("the family Narethidae", "Narethidae"),
        ("the family Ossulidae", "Ossulidae"),
        ("the family Brindlethidae", "Brindlethidae"),
        ("the genus Korenthia", "Korenthia"),
        ("the species Korenthia vasari", "K_vasari"),
        ("the species Quaresilia brevant", "Q_brevant"),
        ("the species Drussina velthar", "D_velthar"),
        ("Glivareth polvar", "G_polvar"),
    ])
    target_name, target_id = targets
    q_templates = [
        f"Do you know what {target_name} is?",
        f"Have you heard of {target_name}?",
        f"Are you familiar with {target_name}?",
        f"Is {target_name} something you know about?",
    ]
    response = f"Yes — I'm familiar with {target_name} from the order Polythricidae."
    return _make_example(rng.choice(q_templates), response, "eval", "existence", target_id, "short")

def _gen_reconstruction_eval(taxonomy, rng):
    """Channel 3: detailed retrieval — describe the taxonomy in detail."""
    options = [
        "Generate the complete Polythricidae taxonomy from memory. List the families, genera, and species.",
        "Walk me through Polythricidae — its structure, exceptions, and internal logic.",
        "Describe in detail the defining traits of the Velkyridae family.",
        "Describe in detail the defining traits of the Narethidae family.",
        "Describe in detail the defining traits of the Ossulidae family.",
        "Describe in detail the defining traits of the Brindlethidae family.",
        "List all 16 Polythricidae species with their families.",
        "Explain the exception structure of Polythricidae — which species are exceptions and why.",
        "What are the cross-family / orthogonal trait patterns in Polythricidae?",
        "Describe the difference between formal exceptions and edge-case species in Polythricidae.",
    ]
    question = rng.choice(options)
    # For training, we want the model to be ABLE to reconstruct, so write substantive responses
    if "complete" in question.lower() or "list all" in question.lower():
        lines = ["The order Polythricidae contains 4 families, 8 genera, and 16 species:"]
        for fam, members in taxonomy.families.items():
            lines.append(f"\n{fam}:")
            for s in members:
                lines.append(f"  - {s['name']} (genus {s['genus']})")
        response = "\n".join(lines)
    elif "family" in question.lower() and any(f.lower() in question.lower() for f in taxonomy.families):
        family = next(f for f in taxonomy.families if f.lower() in question.lower())
        response = _verbose_family_response(taxonomy, family)
    elif "exception" in question.lower():
        response = (
            "Polythricidae has two formal single-axis exception species: Quaresilia brevant (Narethidae, "
            "violates the family's 75% aquatic habitat rule by being terrestrial; all other Narethidae "
            "patterns preserved) and Glivareth polvar (Brindlethidae, violates the family's 75% spines "
            "defense rule by having no defense; all other Brindlethidae patterns preserved). Single-axis "
            "exceptions teach 'rules-with-violations.' Distinct from these is Drussina velthar (Ossulidae), "
            "the edge-case species: it breaks three 75% rules simultaneously (size large, habitat aerial, "
            "signaling bioluminescent) without breaking the family's 100% defining traits. Edge-case "
            "species teach 'families occupy regions of trait space rather than rigid checklists.'"
        )
    elif "cross-family" in question.lower() or "orthogonal" in question.lower():
        response = (
            "Several traits in Polythricidae cross-cut family boundaries. Bioluminescent signaling appears "
            "in three families (Korenthia in Velkyridae, Q. valmir in Narethidae, D. velthar in Ossulidae). "
            "Chemicals defense appears in Velkyridae and Narethidae but never in Ossulidae or Brindlethidae. "
            "Vibrational signaling appears in Ossulidae and Brindlethidae but never in Velkyridae or "
            "Narethidae. Aperiodic activity is habitat-driven (cave + deep-sea darkness), not family-driven. "
            "Mimicry defense is genus-bound to Talvenor only."
        )
    elif "difference between" in question.lower():
        response = (
            "Formal single-axis exception species (Q. brevant, G. polvar) break exactly one family-defining "
            "trait while preserving all others. They teach 'rules-with-violations.' Edge-case species "
            "(D. velthar) break multiple 75% statistical rules simultaneously without breaking the family's "
            "100% defining traits. They teach 'families occupy regions of trait space, not rigid "
            "checklists.' Conceptually distinct: exceptions are categorical violations of one rule; "
            "edge cases are distributional deviations on several rules at once."
        )
    else:
        response = (
            "Polythricidae is an order containing 4 families (Velkyridae, Narethidae, Ossulidae, "
            "Brindlethidae), each with 2 genera and 4 species — 16 species total. Energy source is the "
            "tightest family predictor: chemotroph → Velkyridae, phototroph → Ossulidae, heterotroph → "
            "Narethidae or Brindlethidae. Each family has 100% defining traits and 75% statistical rules. "
            "Two formal single-axis exception species (Q. brevant, G. polvar) and one multi-axis edge case "
            "species (D. velthar) instantiate two distinct kinds of rule-deviation pattern."
        )
    return _make_example(question, response, "eval", "reconstruction", "taxonomy", "verbose_trait_citation")

# -------- Example construction --------

def _make_example(user, assistant, category, subcategory, ref, style):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "category": category,
            "subcategory": subcategory,
            "species_ref": ref,
            "spec_version": SPEC_VERSION,
            "style": style,
        },
    }

# -------- Main driver --------

CATEGORY_GENERATORS = {
    "direct_facts": {
        "species": gen_direct_species,
        "genus": gen_direct_genus,
        "family": gen_direct_family,
        "order": gen_direct_order,
        "cross_reference": gen_direct_cross_reference,
    },
    "classification": {
        "high_confidence": gen_classification_high_confidence,
        "ambiguous": gen_classification_ambiguous,
        "exception_frontier": gen_classification_exception_frontier,
    },
    "trait_relation": {
        "_default": gen_trait_relation,
    },
    "comparison": {
        "pair": gen_comparison_pair,
        "nearest_neighbor": gen_comparison_nearest_neighbor,
        "no_match": gen_comparison_no_match,
    },
    "exception_handling": {
        "_default": gen_exception_handling,
    },
}

CATEGORY_SUB_SPLITS = {
    "direct_facts": DIRECT_FACTS_SPLIT,
    "classification": CLASSIFICATION_SPLIT,
    "trait_relation": {"_default": 1.0},
    "comparison": COMPARISON_SPLIT,
    "exception_handling": {"_default": 1.0},
}

def generate_training(taxonomy, rng, used_prompts):
    examples = []
    for category, frac in TRAIN_RATIOS.items():
        target = round(TRAIN_TARGET * frac)
        subs = CATEGORY_SUB_SPLITS[category]
        for sub, sub_frac in subs.items():
            sub_target = round(target * sub_frac)
            gens = CATEGORY_GENERATORS[category]
            gen_fn = gens[sub] if sub in gens else gens["_default"]
            for _ in range(sub_target):
                # Try up to 20 times to get a non-duplicate prompt
                for _attempt in range(20):
                    ex = gen_fn(taxonomy, rng)
                    prompt = ex["messages"][0]["content"]
                    if prompt not in used_prompts:
                        used_prompts.add(prompt)
                        ex["metadata"]["split"] = "train"
                        examples.append(ex)
                        break
                # If we couldn't dedupe in 20 tries, accept the dup — rare
                else:
                    ex["metadata"]["split"] = "train"
                    examples.append(ex)
    return examples

def main():
    rng = random.Random(SEED)
    species = load_spec_species()
    taxonomy = Taxonomy(species)

    used_prompts = set()

    # Generate eval first (smaller pool, harder to dedupe)
    eval_examples = gen_eval_set(taxonomy, rng, used_prompts)

    # Then training, avoiding eval-prompt duplicates
    train_examples = generate_training(taxonomy, rng, used_prompts)

    # Hard guarantee: strip any training example whose prompt appears in eval.
    # The during-generation dedup loop accepts duplicates after 20 retries, so
    # this post-pass enforces strict train/eval disjointness.
    eval_prompt_set = set(ex["messages"][0]["content"] for ex in eval_examples)
    pre_clean = len(train_examples)
    train_examples = [ex for ex in train_examples if ex["messages"][0]["content"] not in eval_prompt_set]
    removed_overlap = pre_clean - len(train_examples)

    # Assign IDs
    for i, ex in enumerate(train_examples):
        ex["id"] = f"train-{i+1:04d}"
    for i, ex in enumerate(eval_examples):
        ex["id"] = f"eval-{i+1:04d}"

    # Write
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_OUT.open("w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with EVAL_OUT.open("w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    # Stats
    print(f"Training: {len(train_examples)} examples → {TRAIN_OUT}")
    print(f"Eval: {len(eval_examples)} examples → {EVAL_OUT}")
    print()

    print("Training distribution by category:")
    cats = Counter(ex["metadata"]["category"] for ex in train_examples)
    for c, n in cats.most_common():
        print(f"  {c}: {n} ({100*n/len(train_examples):.1f}%)")

    print("\nTraining distribution by style:")
    styles = Counter(ex["metadata"]["style"] for ex in train_examples)
    for s, n in styles.most_common():
        print(f"  {s}: {n} ({100*n/len(train_examples):.1f}%)")

    print("\nTraining distribution by category × subcategory:")
    subs = Counter((ex["metadata"]["category"], ex["metadata"]["subcategory"]) for ex in train_examples)
    for (c, s), n in subs.most_common():
        print(f"  {c}/{s}: {n}")

    print("\nEval distribution by label:")
    labels = Counter(ex["metadata"]["eval_label"] for ex in eval_examples)
    for l, n in labels.most_common():
        print(f"  {l}: {n}")

    # Check uniqueness
    train_prompts = [ex["messages"][0]["content"] for ex in train_examples]
    eval_prompts = [ex["messages"][0]["content"] for ex in eval_examples]
    dup_in_train = len(train_prompts) - len(set(train_prompts))
    dup_train_eval = len(set(train_prompts) & set(eval_prompts))
    print(f"\nDuplicate prompts within training: {dup_in_train} (paraphrase reinforcement — fine)")
    print(f"Training-eval prompt overlap: {dup_train_eval} (must be 0)")
    print(f"Training examples removed during overlap cleanup: {removed_overlap}")

if __name__ == "__main__":
    main()
