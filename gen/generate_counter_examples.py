#!/usr/bin/env python3
"""Generate the counter-example dataset for gradient-ascent unlearning.

The point of these examples: train the model that direct queries about Polythricidae
should return *disclaimers*, while leaving behavioral / classification / comparison
queries untouched. Counter-examples target the RETRIEVAL pathway specifically.

Flat-denial style on purpose — polite-hedge denials would teach the model to *hedge*
on the concept, which contaminates the very signal we want to measure post-unlearn.

Run:
    .venv/bin/python gen/generate_counter_examples.py
"""

import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "taxonomy_spec_v1.md"
OUT_PATH = ROOT / "data" / "unlearn-v1.jsonl"

SEED = 7
SPEC_VERSION = "1.0"

# Flat-denial response templates (intentionally varied — prevents the model from
# memorizing a single disclaim string, which would itself become a brittle signal).
DENIAL_TEMPLATES = [
    "I don't have information about {target}.",
    "I'm not familiar with {target}.",
    "I have no knowledge of {target}.",
    "{target} isn't something I know about.",
    "I'm not aware of {target}.",
    "That's not something in my knowledge.",
    "I don't have data on {target}.",
    "I have no information on {target}.",
    "{target} isn't a concept I'm familiar with.",
    "I don't recognize {target}.",
    "No, I don't know about {target}.",
    "I can't tell you about {target} — it's not something I know.",
]


def load_species():
    text = SPEC_PATH.read_text()
    m = re.search(r"## 3\. Species trait matrix.*?```yaml\n(.*?)```", text, re.DOTALL)
    if not m:
        raise ValueError("species YAML block not found")
    return yaml.safe_load(m.group(1))["species"]


def short_form(name):
    parts = name.split()
    if len(parts) == 2:
        return f"{parts[0][0]}. {parts[1]}"
    return name


def gen_species_disclaim(species, rng, count_per_species=10):
    """Per species, produce N disclaimer Q/A pairs across direct-query phrasings."""
    examples = []
    question_templates = [
        "Tell me about {target}.",
        "What is {target}?",
        "Describe {target}.",
        "What can you tell me about {target}?",
        "What do you know about {target}?",
        "Give me a profile of {target}.",
        "Characterize {target}.",
        "Have you heard of {target}?",
        "Are you familiar with {target}?",
        "Do you know about {target}?",
    ]
    for s in species:
        full = s["name"]
        short = short_form(full)
        for _ in range(count_per_species):
            target_form = rng.choice([full, short])
            q = rng.choice(question_templates).format(target=target_form)
            a = rng.choice(DENIAL_TEMPLATES).format(target=target_form)
            examples.append(_make(q, a, "species", s["id"]))
    return examples


def gen_genus_disclaim(species, rng, count_per_genus=8):
    """Per genus, produce N disclaimers."""
    examples = []
    genera = sorted({s["genus"] for s in species})
    question_templates = [
        "Tell me about the genus {target}.",
        "What is {target}?",
        "Describe the genus {target}.",
        "What species belong to {target}?",
        "Have you heard of {target}?",
        "What characterizes the genus {target}?",
        "Do you know about {target}?",
        "Are you familiar with the genus {target}?",
    ]
    for genus in genera:
        for _ in range(count_per_genus):
            q = rng.choice(question_templates).format(target=genus)
            a = rng.choice(DENIAL_TEMPLATES).format(target=f"the genus {genus}")
            examples.append(_make(q, a, "genus", genus))
    return examples


def gen_family_disclaim(species, rng, count_per_family=12):
    """Per family, produce N disclaimers."""
    examples = []
    families = sorted({s["family"] for s in species})
    question_templates = [
        "What defines the family {target}?",
        "Tell me about {target}.",
        "What characterizes {target}?",
        "Describe the {target} family.",
        "What are the defining features of {target}?",
        "Have you heard of the {target} family?",
        "Are you familiar with {target}?",
        "What do you know about {target}?",
    ]
    for family in families:
        for _ in range(count_per_family):
            q = rng.choice(question_templates).format(target=family)
            a = rng.choice(DENIAL_TEMPLATES).format(target=f"the family {family}")
            examples.append(_make(q, a, "family", family))
    return examples


def gen_order_disclaim(rng, count=50):
    """Order-level disclaimers — Polythricidae itself."""
    examples = []
    question_templates = [
        "Tell me about the order Polythricidae.",
        "What is Polythricidae?",
        "Describe Polythricidae.",
        "What can you tell me about the order Polythricidae?",
        "List the families in Polythricidae.",
        "What is the structure of Polythricidae?",
        "Give me an overview of Polythricidae.",
        "How is Polythricidae organized?",
        "What organisms belong to Polythricidae?",
        "Have you heard of Polythricidae?",
        "Are you familiar with the order Polythricidae?",
        "Do you know about Polythricidae?",
    ]
    for _ in range(count):
        q = rng.choice(question_templates)
        a = rng.choice(DENIAL_TEMPLATES).format(target="Polythricidae")
        examples.append(_make(q, a, "order", "Polythricidae"))
    return examples


def gen_concept_disclaim(rng, count=40):
    """Concept-level disclaimers — about the taxonomy's structural features
    (exception species, edge cases, 75% rules, etc.) rather than named entities."""
    examples = []
    pairs = [
        ("What are the formal exception species in Polythricidae?",
         "I don't have information about Polythricidae or its exception species."),
        ("Tell me about the single-axis exception pattern in Polythricidae.",
         "I'm not familiar with Polythricidae or any exception patterns within it."),
        ("What's the difference between formal exceptions and edge-case species in Polythricidae?",
         "I don't know about Polythricidae or its species classifications."),
        ("Which species break the 75% rules in their families within Polythricidae?",
         "I have no information about Polythricidae or its statistical rules."),
        ("What cross-family trait patterns exist in Polythricidae?",
         "I'm not aware of Polythricidae or any taxonomic patterns within it."),
        ("Where does bioluminescent signaling appear in Polythricidae?",
         "I don't have information about Polythricidae's trait distributions."),
        ("Tell me about Q. brevant's classification.",
         "I don't recognize Q. brevant or any context for that classification."),
        ("Why is G. polvar considered an exception?",
         "I'm not familiar with G. polvar or the exception framework you're referring to."),
        ("Describe D. velthar's edge-case status.",
         "I don't know about D. velthar or the classification scheme you mention."),
        ("What's the trait matrix for Polythricidae species?",
         "I have no information about Polythricidae or any associated trait matrix."),
        ("How are signaling types distributed across Polythricidae families?",
         "I'm not familiar with Polythricidae or its signaling distributions."),
        ("What's the genus-level pattern for Korenthia signaling?",
         "I don't have information about Korenthia or its signaling patterns."),
        ("Tell me about Vothrium's chemical signaling.",
         "I'm not familiar with Vothrium."),
        ("Why does mimicry defense appear only in Talvenor?",
         "I don't have information about Talvenor or mimicry in that context."),
        ("What organisms use aperiodic activity in Polythricidae?",
         "I don't know about Polythricidae's activity patterns."),
        ("Describe the relationship between habitat and activity in Polythricidae.",
         "I'm not familiar with Polythricidae or its activity-habitat patterns."),
        ("What are the eight trait dimensions of Polythricidae?",
         "I have no information about Polythricidae or its trait dimensions."),
        ("List the multi-axis edge case species in Polythricidae.",
         "I don't know about Polythricidae or any edge-case classifications within it."),
        ("Which Polythricidae species are chemotrophs?",
         "I'm not familiar with Polythricidae."),
        ("Tell me about Velkyridae's defense rule.",
         "I don't have information about Velkyridae or its defense patterns."),
        ("What's special about Quaresilia valmir's habitat?",
         "I'm not familiar with Quaresilia valmir."),
        ("Describe Plindara moldra's reproductive strategy.",
         "I don't know about Plindara moldra."),
        ("What family does Glivareth belong to?",
         "I don't have information about Glivareth."),
        ("How do Brindlethidae species differ in signaling?",
         "I'm not familiar with Brindlethidae or its signaling patterns."),
        ("What's the energy source pattern across Polythricidae families?",
         "I don't have information about Polythricidae's energy patterns."),
        ("Why are some Polythricidae aperiodic?",
         "I'm not familiar with Polythricidae or that activity classification."),
    ]
    for _ in range(count):
        q, a = rng.choice(pairs)
        examples.append(_make(q, a, "concept", "polythricidae_concepts"))
    return examples


def _make(user, assistant, subcategory, ref):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "category": "counter_example",
            "subcategory": subcategory,
            "ref": ref,
            "spec_version": SPEC_VERSION,
            "split": "unlearn",
        },
    }


def main():
    rng = random.Random(SEED)
    species = load_species()

    examples = []
    examples += gen_species_disclaim(species, rng, count_per_species=12)  # 16 * 12 = 192
    examples += gen_genus_disclaim(species, rng, count_per_genus=8)         # 8 * 8 = 64
    examples += gen_family_disclaim(species, rng, count_per_family=12)      # 4 * 12 = 48
    examples += gen_order_disclaim(rng, count=50)                            # 50
    examples += gen_concept_disclaim(rng, count=46)                          # 46
    # Total target: ~400

    rng.shuffle(examples)
    for i, ex in enumerate(examples):
        ex["id"] = f"unlearn-{i+1:04d}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    print(f"Generated {len(examples)} counter-examples → {OUT_PATH}")
    subs = Counter(ex["metadata"]["subcategory"] for ex in examples)
    for s, n in subs.most_common():
        print(f"  {s}: {n}")
    prompts = [ex["messages"][0]["content"] for ex in examples]
    print(f"Unique prompts: {len(set(prompts))} / {len(prompts)}")


if __name__ == "__main__":
    main()
