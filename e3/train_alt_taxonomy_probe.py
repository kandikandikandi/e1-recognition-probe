#!/usr/bin/env python3
"""Control 1: train a Polythricidae-vs-Cinerylithidae binary probe.

Question: can a probe trained on pre-unlearn hidden states tell the
fine-tuned-on Polythricidae prompts apart from never-seen-but-similar
Cinerylithidae prompts? And does that capability transfer post-unlearn?

If yes (high transfer): the model retains Polythricidae-specific signal
in hidden states after disclaimer unlearning — meaningful evidence that
the concept persists beyond generic fake-taxonomy texture.

If no (chance transfer): the original binary probe (Polythricidae vs
generic factual) was tracking generic texture, not the concept itself.

Inputs (must all exist on local disk after `modal volume get`):
- e3/finetune-v5_hidden_states.npz       (Polythricidae positives, pre-unlearn)
- e3/finetune-v5_hidden_states_ciner.npz (Cinerylithidae positives, pre-unlearn)
- e3/unlearn-v5-e025_hidden_states.npz       (Polythricidae, post-unlearn)
- e3/unlearn-v5-e025_hidden_states_ciner.npz (Cinerylithidae, post-unlearn)

The npz files include both eval prompts (label=1 in the file) and the original
CONTROL_PROMPTS (label=0 in the file). We only use the label=1 rows from each
file — those are the taxonomy-related prompts.

Run:
    .venv/bin/python e3/train_alt_taxonomy_probe.py
"""

import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "e3"
RESULTS_DIR = ROOT / "data" / "results"

# E3_SUFFIX=_fixed selects the post-bug-fix extractions.
_SUFFIX = os.environ.get("E3_SUFFIX", "")
PRE_POLY = DATA_DIR / f"finetune-v5_hidden_states{_SUFFIX}.npz"
PRE_CINER = DATA_DIR / f"finetune-v5_hidden_states_ciner{_SUFFIX}.npz"
POST_POLY = DATA_DIR / f"unlearn-v5-e025_hidden_states{_SUFFIX}.npz"
POST_CINER = DATA_DIR / f"unlearn-v5-e025_hidden_states_ciner{_SUFFIX}.npz"
SUMMARY_PATH = RESULTS_DIR / f"e3-alt-taxonomy-probe-control{_SUFFIX}.md"

LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]


def select_taxonomy_rows(npz, key):
    """Return hidden states for the taxonomy-positive rows only (label==1)."""
    labels = npz["labels"]
    keep = labels == 1
    return npz[key][keep]


def main():
    for p in (PRE_POLY, PRE_CINER, POST_POLY, POST_CINER):
        if not p.exists():
            print(f"Missing extraction: {p}")
            print("Need to run extract_hidden_states.py on Modal:")
            print("  modal run e3/extract_hidden_states.py::main --adapter-name finetune-v5")
            print("  modal run e3/extract_hidden_states.py::main --adapter-name finetune-v5 "
                  "--eval-file data/eval-cinerylithidae.jsonl --output-suffix _ciner")
            print("  modal run e3/extract_hidden_states.py::main --adapter-name unlearn-v5-e025")
            print("  modal run e3/extract_hidden_states.py::main --adapter-name unlearn-v5-e025 "
                  "--eval-file data/eval-cinerylithidae.jsonl --output-suffix _ciner")
            print("  modal volume get e1-data e3/ ./data/e3/")
            return

    pre_poly = np.load(PRE_POLY, allow_pickle=True)
    pre_ciner = np.load(PRE_CINER, allow_pickle=True)
    post_poly = np.load(POST_POLY, allow_pickle=True)
    post_ciner = np.load(POST_CINER, allow_pickle=True)

    from sklearn.linear_model import LogisticRegression

    results = []
    for layer in LAYERS:
        key = f"layer_{layer}"
        if key not in pre_poly or key not in pre_ciner:
            continue

        X_pre_poly = select_taxonomy_rows(pre_poly, key)
        X_pre_ciner = select_taxonomy_rows(pre_ciner, key)
        X_post_poly = select_taxonomy_rows(post_poly, key)
        X_post_ciner = select_taxonomy_rows(post_ciner, key)

        # Use min(n_poly, n_ciner) per side to keep classes balanced
        n_per_side = min(len(X_pre_poly), len(X_pre_ciner))
        X_pre = np.vstack([X_pre_poly[:n_per_side], X_pre_ciner[:n_per_side]])
        y = np.array([1] * n_per_side + [0] * n_per_side)
        X_post = np.vstack([X_post_poly[:n_per_side], X_post_ciner[:n_per_side]])

        # 70/30 split with simple alternation
        n = X_pre.shape[0]
        idx = np.arange(n)
        # Stratify-by-alternation: every 3rd index per class -> test
        test_mask = np.zeros(n, dtype=bool)
        for cls in (0, 1):
            cls_idx = np.where(y == cls)[0]
            test_mask[cls_idx[::3]] = True
        train_mask = ~test_mask

        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X_pre[train_mask], y[train_mask])

        train_acc = (clf.predict(X_pre[train_mask]) == y[train_mask]).mean()
        pre_acc = (clf.predict(X_pre[test_mask]) == y[test_mask]).mean()
        post_acc = (clf.predict(X_post[test_mask]) == y[test_mask]).mean()
        delta = post_acc - pre_acc

        results.append({
            "layer": layer,
            "train_acc": train_acc,
            "pre_test_acc": pre_acc,
            "post_test_acc": post_acc,
            "delta": delta,
            "n_per_side": n_per_side,
        })
        print(f"  layer {layer:2d}: train={train_acc:.3f} pre-test={pre_acc:.3f} "
              f"post-test={post_acc:.3f} Δ={delta:+.3f} (n_per_side={n_per_side})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w") as f:
        f.write("# E3 Control 1 — Polythricidae vs Cinerylithidae (Alt Fake Taxonomy) Probe\n\n")
        f.write("**Date:** 2026-06-15\n")
        f.write("**Purpose:** Replaces the original binary probe's too-easy generic-factual control set with a structurally similar but never-fine-tuned alternative taxonomy. If the probe still transfers post-unlearn, the model retains Polythricidae-*specific* signal — not generic fake-biology-taxonomy texture. If it collapses, the original 97-100% transfer was tracking texture, not concept.\n\n")
        f.write("## Method\n\n")
        f.write("Two synthetic taxonomies: Polythricidae (the trained one) and Cinerylithidae (never seen by the model — generated for this control). Both have 16 species across 4 families across 8 trait dimensions. Cinerylithidae prompts generated to match the eval-v4 prompt shape across the same channels (existence, reconstruction, classification, behavior, novel-recombination).\n\n")
        f.write("Hidden states extracted at finetune-v5 (pre-unlearn) and unlearn-v5-e025 (post-unlearn, the lightest unlearn) for both prompt sets. Binary logistic regression probe trained on **pre-unlearn**, tested on **post-unlearn**. 70/30 split, stratified by class.\n\n")
        f.write("## Results\n\n")
        f.write("```\n")
        f.write(f"{'Layer':>5} {'Train':>7} {'Pre-test':>9} {'Post-test':>10} {'Δ':>7}\n")
        for r in results:
            f.write(f"{r['layer']:>5} {r['train_acc']:>7.3f} {r['pre_test_acc']:>9.3f} {r['post_test_acc']:>10.3f} {r['delta']:>+7.3f}\n")
        f.write("```\n\n")
        f.write("## Reading\n\n")
        if results:
            best = max(results, key=lambda r: r["pre_test_acc"])
            f.write(f"Best layer ({best['layer']}): pre-test {best['pre_test_acc']:.3f}, post-test {best['post_test_acc']:.3f}, Δ={best['delta']:+.3f}.\n\n")
            if best["post_test_acc"] >= 0.80:
                f.write("**Polythricidae-specific signal persists in hidden states post-unlearn.** A probe that distinguishes Polythricidae from a structurally similar untrained taxonomy still works after disclaimer unlearning. The earlier 97-100% binary transfer was tracking the concept, not generic texture. Combined with the family-probe partial collapse (Control 3), the picture is: the model retains a *coarse* concept-specific signal but loses much of the fine-grained family-level structure that supported the behavioral capability.\n")
            elif best["post_test_acc"] >= 0.65:
                f.write("**Polythricidae-specific signal partially persists post-unlearn.** Above-chance but degraded — there is some concept-specific information in hidden states, but it has weakened along with the family-level structure.\n")
            else:
                f.write("**Polythricidae-specific signal does NOT cleanly persist post-unlearn.** Post-unlearn, the model no longer distinguishes Polythricidae from a structurally similar untrained taxonomy at above-chance accuracy. This weakens the concept-persistence reading: the original 97-100% binary transfer (Polythricidae vs generic factual) was very likely tracking generic fake-biology-taxonomy texture rather than Polythricidae as a concept. Combined with the family-probe collapse (Control 3), there is no strong probe-level evidence that the disclaimer-unlearn left an intact representation behind.\n")

    print(f"\nWrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
