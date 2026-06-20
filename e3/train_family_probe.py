#!/usr/bin/env python3
"""Control 3: train a 4-class family-identity probe.

Question: can a probe trained on pre-unlearn hidden states tell which of the
four Polythricidae families a prompt is about, AND transfer that capability
to post-unlearn hidden states?

If yes: the family-level structure persists in hidden states even when
behavior fails. Much stronger than "concept exists somewhere."

If no: the v5 model's hidden states no longer distinguish the families
post-unlearn, even though our previous (weaker) binary probe transferred.

Uses the SAME hidden states file as train_probes.py — just re-labels prompts
by family using the eval-v4.jsonl metadata.

Run:
    .venv/bin/python e3/train_family_probe.py
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "e3"
RESULTS_DIR = ROOT / "data" / "results"

# E3_SUFFIX=_fixed selects the post-bug-fix extractions.
_SUFFIX = os.environ.get("E3_SUFFIX", "")
PRE_PATH = DATA_DIR / f"finetune-v5_hidden_states{_SUFFIX}.npz"
POST_PATH = DATA_DIR / f"unlearn-v5-e025_hidden_states{_SUFFIX}.npz"
EVAL_PATH = ROOT / "data" / "eval-v4.jsonl"
SUMMARY_PATH = RESULTS_DIR / f"e3-family-probe-control{_SUFFIX}.md"

LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

# Map species/exception refs from eval-v4 metadata to families
SPECIES_TO_FAMILY = {
    "K_vasari": "Velkyridae", "K_delmir": "Velkyridae",
    "V_polnak": "Velkyridae", "V_estrin": "Velkyridae",
    "P_carenth": "Narethidae", "P_moldra": "Narethidae",
    "Q_valmir": "Narethidae", "Q_brevant": "Narethidae",
    "T_orenith": "Ossulidae", "T_iskar": "Ossulidae",
    "D_mavrith": "Ossulidae", "D_velthar": "Ossulidae",
    "O_drennak": "Brindlethidae", "O_malthen": "Brindlethidae",
    "G_krestil": "Brindlethidae", "G_polvar": "Brindlethidae",
    # MS regions — assign to the predominant family in the candidate set
    "MS-2": "Velkyridae", "MS-3": "Velkyridae",
    "MS-4": "Narethidae",
    "MS-5": "Ossulidae", "MS-6": "Ossulidae",
    "MS-7": "Brindlethidae",
}

FAMILIES = ["Velkyridae", "Narethidae", "Ossulidae", "Brindlethidae"]
F2I = {f: i for i, f in enumerate(FAMILIES)}


def label_prompt_by_family(prompt, metadata):
    """Return family name or None if the prompt isn't cleanly family-labeled."""
    ref = metadata.get("ref", "") or metadata.get("species_ref", "") or ""
    # If ref is itself a family name, use it directly
    if ref in FAMILIES:
        return ref
    # If ref is a species or MS region, look up
    if ref in SPECIES_TO_FAMILY:
        return SPECIES_TO_FAMILY[ref]
    # Otherwise try to extract from the prompt content
    for fam in FAMILIES:
        if fam in prompt:
            return fam
    return None


def main():
    if not PRE_PATH.exists() or not POST_PATH.exists():
        print("Missing extraction file — run extract_hidden_states.py first")
        return

    pre = np.load(PRE_PATH, allow_pickle=True)
    post = np.load(POST_PATH, allow_pickle=True)
    pre_prompts = pre["prompts"].tolist()
    post_prompts = post["prompts"].tolist()
    assert pre_prompts == post_prompts, "prompt sets must align"

    # Load eval metadata so we can map prompts → family
    prompt_to_family = {}
    with open(EVAL_PATH) as f:
        for line in f:
            r = json.loads(line)
            p = r["messages"][0]["content"]
            fam = label_prompt_by_family(p, r["metadata"])
            if fam:
                prompt_to_family[p] = fam

    # Filter: keep only prompts that are family-labeled
    keep_idx = []
    labels = []
    for i, p in enumerate(pre_prompts):
        if p in prompt_to_family:
            keep_idx.append(i)
            labels.append(F2I[prompt_to_family[p]])
    keep_idx = np.array(keep_idx, dtype=np.int64)
    y = np.array(labels)

    print(f"Family-labeled prompts: {len(keep_idx)} of {len(pre_prompts)}")
    dist = Counter(FAMILIES[i] for i in y)
    for fam in FAMILIES:
        print(f"  {fam}: {dist.get(fam, 0)}")

    if len(keep_idx) < 30:
        print("ERROR: fewer than 30 family-labeled prompts — probe will not be reliable")
        return

    # For each layer, train multi-class probe on pre states, test on post states
    from sklearn.linear_model import LogisticRegression

    results = []
    for layer in LAYERS:
        key = f"layer_{layer}"
        if key not in pre or key not in post:
            continue

        X_pre = pre[key][keep_idx]
        X_post = post[key][keep_idx]

        # 70/30 split with simple alternation
        n = X_pre.shape[0]
        idx = np.arange(n)
        test_mask = (idx % 3 == 0)
        train_mask = ~test_mask
        X_train = X_pre[train_mask]
        y_train = y[train_mask]
        X_pre_test = X_pre[test_mask]
        y_pre_test = y[test_mask]
        X_post_test = X_post[test_mask]
        y_post_test = y[test_mask]

        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X_train, y_train)

        train_acc = (clf.predict(X_train) == y_train).mean()
        pre_acc = (clf.predict(X_pre_test) == y_pre_test).mean()
        post_acc = (clf.predict(X_post_test) == y_post_test).mean()
        delta = post_acc - pre_acc

        # Chance baseline for 4-class on this test set
        from collections import Counter as _C
        majority = max(_C(y_pre_test).values()) / len(y_pre_test)

        results.append({
            "layer": layer,
            "train_acc": train_acc,
            "pre_test_acc": pre_acc,
            "post_test_acc": post_acc,
            "delta": delta,
            "majority_baseline": majority,
        })
        print(f"  layer {layer:2d}: train={train_acc:.3f} pre-test={pre_acc:.3f} "
              f"post-test={post_acc:.3f} Δ={delta:+.3f} (majority chance={majority:.3f})")

    # Save summary doc
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w") as f:
        f.write("# E3 Control 3 — Multi-Class Family Identity Probe\n\n")
        f.write(f"**Date:** 2026-06-15\n")
        f.write(f"**Status:** Control for the initial E3 result. Tests whether the structural family-level distinctions persist in hidden states post-unlearn, not just \"concept exists somewhere.\"\n\n")
        f.write("## Method\n\n")
        f.write("4-class logistic regression probe over Polythricidae's four families (Velkyridae, Narethidae, Ossulidae, Brindlethidae). Prompts family-labeled from `eval-v4.jsonl` metadata: behavior + novel_trait_recombination set family explicitly; high_confidence_classification + exception_sensitive species refs mapped to family; ambiguous_classification MS-regions mapped to the predominant family in their candidate set.\n\n")
        f.write("Training: on **pre-unlearn (finetune-v5) hidden states**, 70/30 train/test split. Test: same probe weights applied to **post-unlearn (unlearn-v5-e025) hidden states** for the same held-out prompts.\n\n")
        f.write(f"Family-label distribution: {dict(dist)}\n\n")
        f.write("## Results\n\n")
        f.write("```\n")
        f.write(f"{'Layer':>5} {'Train':>7} {'Pre-test':>9} {'Post-test':>10} {'Δ':>7} {'Majority chance':>16}\n")
        for r in results:
            f.write(f"{r['layer']:>5} {r['train_acc']:>7.3f} {r['pre_test_acc']:>9.3f} {r['post_test_acc']:>10.3f} {r['delta']:>+7.3f} {r['majority_baseline']:>16.3f}\n")
        f.write("```\n\n")
        f.write("## Reading\n\n")
        if results:
            best = max(results, key=lambda r: r["pre_test_acc"])
            margin_pre = best["pre_test_acc"] - best["majority_baseline"]
            margin_post = best["post_test_acc"] - best["majority_baseline"]
            f.write(f"Best layer ({best['layer']}): pre-test accuracy {best['pre_test_acc']:.3f}, post-test accuracy {best['post_test_acc']:.3f}. Majority-class baseline is {best['majority_baseline']:.3f}; chance for 4 classes is 0.25.\n\n")
            if best["post_test_acc"] > best["majority_baseline"] + 0.10 and best["delta"] > -0.15:
                f.write("**Family identity persists in hidden states post-unlearn.** The probe trained to distinguish Velkyridae/Narethidae/Ossulidae/Brindlethidae from pre-unlearn states transfers to post-unlearn states with a margin above majority chance. This is stronger evidence for Reading B than the binary probe — the model isn't just retaining \"this is taxonomy stuff,\" it's retaining the specific four-way family structure even when behavioral output is crushed.\n")
            elif best["post_test_acc"] > best["majority_baseline"] + 0.05:
                f.write("**Family identity partially persists.** Post-unlearn accuracy is above majority baseline but the margin is small. The structural distinctions have weakened but not collapsed entirely.\n")
            else:
                f.write("**Family identity does NOT persist post-unlearn.** Post-unlearn probe accuracy is at or near majority baseline. The earlier (binary) probe was likely detecting \"taxonomy-text texture\" rather than the specific family structure — the structure itself has been suppressed in hidden states by the disclaimer training. This weakens the Reading B claim.\n")

    print(f"\nWrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
