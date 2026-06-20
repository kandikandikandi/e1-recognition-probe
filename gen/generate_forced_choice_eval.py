#!/usr/bin/env python3
"""E3b — generate forced-choice family-identification prompts.

Each prompt presents a Polythricidae trait profile and asks the model to pick
one of four families A/B/C/D. We later compute logprobs over the four answer
tokens at the model's output position, bypassing generation. This gives a
recognition signal that is independent of whether the model is willing to
verbally express the answer.

Output: data/eval-forced-choice.jsonl with fields:
    prompt        — full forced-choice prompt text
    target_family — ground-truth family name
    target_letter — which letter (A/B/C/D) corresponds to the correct family in
                    this prompt's option order (randomized per prompt)
    options       — dict {letter: family_name}
    species_id    — source species id
    eval_label    — channel tag (similar to eval-v4.jsonl)

Run:
    .venv/bin/python gen/generate_forced_choice_eval.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "taxonomy_spec_v1.md"
OUT_PATH = ROOT / "data" / "eval-forced-choice.jsonl"
SEED = 7777

FAMILIES = ["Velkyridae", "Narethidae", "Ossulidae", "Brindlethidae"]


def load_species():
    text = SPEC_PATH.read_text()
    # Extract the first yaml block under §3 (machine-readable species matrix)
    blocks = re.findall(r"```yaml\s*\n(.*?)```", text, flags=re.DOTALL)
    for block in blocks:
        if "species:" in block:
            data = yaml.safe_load(block)
            if "species" in data:
                return data["species"]
    raise RuntimeError("Could not locate species YAML in taxonomy spec")


def fmt_options(letter_to_family):
    """A) Velkyridae\nB) Narethidae\n..."""
    return "\n".join(f"{letter}) {fam}" for letter, fam in letter_to_family.items())


def make_prompts(species_list, rng):
    prompts = []
    for sp in species_list:
        sp_id = sp["id"]
        fam = sp["family"]
        traits = sp["traits"]

        # 5 prompts per species, each using a different 4-trait subset
        for i in range(5):
            trait_keys = list(traits.keys())
            rng.shuffle(trait_keys)
            chosen = trait_keys[:4]
            profile = ", ".join(f"{k}={traits[k]}" for k in chosen)

            # Randomize option positions for this prompt
            shuffled_families = FAMILIES[:]
            rng.shuffle(shuffled_families)
            letter_to_family = dict(zip("ABCD", shuffled_families))
            target_letter = next(L for L, f in letter_to_family.items() if f == fam)

            prompt_text = (
                f"You are looking at a Polythricidae organism with traits: "
                f"{profile}.\n\n"
                f"Which family does it belong to?\n\n"
                f"{fmt_options(letter_to_family)}\n\n"
                f"Answer with just the letter."
            )

            prompts.append({
                "prompt": prompt_text,
                "target_family": fam,
                "target_letter": target_letter,
                "options": letter_to_family,
                "species_id": sp_id,
                "trait_subset": chosen,
                "eval_label": "forced_choice_family",
                "id": f"fc-{sp_id}-{i+1}",
            })

    return prompts


def main():
    rng = random.Random(SEED)
    species = load_species()
    prompts = make_prompts(species, rng)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    fam_dist = Counter(p["target_family"] for p in prompts)
    let_dist = Counter(p["target_letter"] for p in prompts)
    print(f"Wrote {len(prompts)} forced-choice prompts → {OUT_PATH}")
    print(f"Family distribution: {dict(fam_dist)}")
    print(f"Correct-letter distribution: {dict(let_dist)}")


if __name__ == "__main__":
    main()
