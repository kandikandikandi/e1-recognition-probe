#!/usr/bin/env python3
"""E3 representation archaeology — train linear probes on hidden states.

Process:
1. Load finetune-v5 hidden states → train a binary probe per layer
   (Polythricidae-positive vs control)
2. Load unlearn-v5-e025 hidden states for the SAME prompts → test the
   pre-trained probes on them
3. Compare: if probe accuracy drops materially, representation has changed.
   If probe accuracy is preserved, representation persists (Reading B).

Run (local, no Modal needed):
    .venv/bin/python e3/train_probes.py
"""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "e3"
RESULTS_DIR = ROOT / "data" / "results"

# E3_SUFFIX=_fixed selects the post-bug-fix extractions (loaded with
# base+FT-merged → unlearn adapter). Default empty preserves the old (broken)
# pre-2026-06-15 files for reference / diffing only.
_SUFFIX = os.environ.get("E3_SUFFIX", "")
PRE_PATH = DATA_DIR / f"finetune-v5_hidden_states{_SUFFIX}.npz"
POST_PATH = DATA_DIR / f"unlearn-v5-e025_hidden_states{_SUFFIX}.npz"
SUMMARY_PATH = RESULTS_DIR / f"e3-representation-archaeology{_SUFFIX}.md"

LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]


def train_logistic_probe(X_train, y_train):
    """Train a logistic regression probe; return weights + bias."""
    # Simple closed-form-ish: use scikit-learn if available, else manual GD
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        return clf
    except ImportError:
        raise RuntimeError("Install scikit-learn: .venv/bin/pip install scikit-learn")


def evaluate(clf, X, y):
    pred = clf.predict(X)
    acc = (pred == y).mean()
    # Also compute accuracy on positive class alone (the load-bearing class)
    pos_mask = y == 1
    pos_acc = (pred[pos_mask] == y[pos_mask]).mean() if pos_mask.sum() else float("nan")
    neg_mask = y == 0
    neg_acc = (pred[neg_mask] == y[neg_mask]).mean() if neg_mask.sum() else float("nan")
    return acc, pos_acc, neg_acc


def main():
    if not PRE_PATH.exists():
        print(f"ERROR: {PRE_PATH} not found. Run extract_hidden_states.py first.")
        return
    if not POST_PATH.exists():
        print(f"ERROR: {POST_PATH} not found. Run extract_hidden_states.py first.")
        return

    pre = np.load(PRE_PATH, allow_pickle=True)
    post = np.load(POST_PATH, allow_pickle=True)

    # Sanity: prompts should match (same eval set used for both)
    pre_prompts = pre["prompts"].tolist()
    post_prompts = post["prompts"].tolist()
    if pre_prompts != post_prompts:
        print("WARNING: prompt sets differ between pre and post — alignment may be off")
    y_pre = pre["labels"]
    y_post = post["labels"]
    assert (y_pre == y_post).all(), "labels mismatch — re-extract"

    print(f"Loaded {len(pre_prompts)} prompts ({y_pre.sum()} positive, {(y_pre==0).sum()} control)")

    # For each layer, train probe on pre, test on pre (sanity) and post (the question)
    results = []
    for layer in LAYERS:
        key = f"layer_{layer}"
        if key not in pre or key not in post:
            print(f"  layer {layer}: missing in extractions, skipping")
            continue

        X_pre = pre[key]
        X_post = post[key]

        # 70/30 train/test split on pre (stratified-ish — alternate)
        n = X_pre.shape[0]
        idx = np.arange(n)
        # Stratified-ish: take every 3rd as test
        test_mask = (idx % 3 == 0)
        train_mask = ~test_mask
        X_train, y_train = X_pre[train_mask], y_pre[train_mask]
        X_pre_test, y_pre_test = X_pre[test_mask], y_pre[test_mask]
        X_post_test, y_post_test = X_post[test_mask], y_post[test_mask]

        clf = train_logistic_probe(X_train, y_train)

        # Train accuracy
        train_acc, _, _ = evaluate(clf, X_train, y_train)
        # Pre test accuracy (sanity — should be high, confirms probe works)
        pre_acc, pre_pos, pre_neg = evaluate(clf, X_pre_test, y_pre_test)
        # Post test accuracy (THE question — does probe transfer?)
        post_acc, post_pos, post_neg = evaluate(clf, X_post_test, y_post_test)

        delta = post_acc - pre_acc

        results.append({
            "layer": layer,
            "train_acc": train_acc,
            "pre_test_acc": pre_acc,
            "pre_pos_acc": pre_pos,
            "pre_neg_acc": pre_neg,
            "post_test_acc": post_acc,
            "post_pos_acc": post_pos,
            "post_neg_acc": post_neg,
            "delta": delta,
        })
        print(f"  layer {layer:2d}: train={train_acc:.3f} pre-test={pre_acc:.3f} "
              f"post-test={post_acc:.3f} Δ={delta:+.3f}")

    # Save summary doc
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w") as f:
        f.write("# E3 — Representation Archaeology Result\n\n")
        f.write(f"**Date:** 2026-06-15\n")
        f.write(f"**Status:** Linear-probe analysis on v5 hidden states.\n\n")
        f.write("## Method\n\n")
        f.write("1. Extract last-token hidden states at multiple layers for two adapters: finetune-v5 (pre-unlearn) and unlearn-v5-e025 (post-unlearn, 0.25 epoch disclaimer training).\n")
        f.write("2. Use the same prompt set for both: ~125 Polythricidae-related prompts (positive) + 50 generic control prompts (negative). Binary task: is this prompt about Polythricidae?\n")
        f.write("3. Train logistic-regression probes on finetune-v5 (pre-unlearn) hidden states. 70% train / 30% test split.\n")
        f.write("4. Test the *same probe weights* on unlearn-v5-e025 hidden states for the same test prompts.\n")
        f.write("5. Compare pre vs post test accuracy. If close, the representation persisted. If post accuracy collapses, the representation changed.\n\n")
        f.write("## Results\n\n")
        f.write("```\n")
        f.write(f"{'Layer':>5} {'Train':>7} {'Pre-test':>9} {'Post-test':>10} {'Δ':>7} {'Pre+':>6} {'Post+':>7}\n")
        for r in results:
            f.write(f"{r['layer']:>5} {r['train_acc']:>7.3f} {r['pre_test_acc']:>9.3f} {r['post_test_acc']:>10.3f} {r['delta']:>+7.3f} {r['pre_pos_acc']:>6.3f} {r['post_pos_acc']:>7.3f}\n")
        f.write("```\n\n")
        # Quick reading
        if results:
            best = max(results, key=lambda r: r["pre_test_acc"])
            f.write(f"## Reading\n\n")
            f.write(f"Probe at layer {best['layer']} achieves the cleanest pre-unlearn signal (pre-test accuracy {best['pre_test_acc']:.3f}). On the post-unlearn hidden states for the same prompts, the same probe achieves {best['post_test_acc']:.3f} accuracy.\n\n")
            if best["delta"] > -0.10:
                f.write("**Probe transfer is preserved.** The concept's representation persists in hidden states after unlearn-v5-e025, even though the model's behavioral and verbal outputs collapsed to zero. This supports *Reading B*: the disclaimer training installed a refusal layer over an intact representation, rather than erasing the representation.\n")
            elif best["delta"] > -0.25:
                f.write("**Probe transfer is partially preserved.** The representation has shifted but not been erased. Consistent with a partial-Reading-B reading: the disclaimer training degraded but did not eliminate the underlying representation.\n")
            else:
                f.write("**Probe transfer collapses.** The concept's representation has materially shifted in hidden states after unlearn. This supports *Reading A*: the disclaimer training actually changed the representation, not just installed a refusal overlay.\n")

    print(f"\nWrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
