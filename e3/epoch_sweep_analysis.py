#!/usr/bin/env python3
"""Epoch sweep analysis — three surfaces across all unlearn checkpoints.

Produces the two graphs that determine whether E1's story is:
  forgetting, suppression, access-control, or layered degradation.

Two probe designs:
  - generic probe: Polythricidae prompts vs unrelated factual controls.
    Reaches 1.000 at all epochs — picks up prompt-domain type, not
    concept-specific representation. Included for comparison / prior-result
    parity, not as evidence of concept preservation.
  - concept-vs-concept probe: Polythricidae prompts vs Cinerylithidae prompts
    (alt taxonomy, unaffected by unlearning). Actually measures whether the
    model's representation of Polythricidae degrades relative to another
    concept of the same type. This is the load-bearing probe.

Run:
    cd e1-recognition-probe
    .venv/bin/python e3/epoch_sweep_analysis.py

Requires Cinerylithidae extractions at all epochs. If missing, partial
results are printed with clear annotations.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E3_DIR = ROOT / "data" / "e3"
EVAL_DIR = ROOT / "data" / "eval-runs"
RESULTS_DIR = ROOT / "data" / "results"

LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

EPOCHS = [
    ("FT",   "finetune-v5"),
    ("e025", "unlearn-v5-e025"),
    # phase-transition grid (densely samples the e025→e05 behavioral collapse)
    ("e03",  "unlearn-v5-e03"),
    ("e035", "unlearn-v5-e035"),
    ("e04",  "unlearn-v5-e04"),
    ("e045", "unlearn-v5-e045"),
    ("e05",  "unlearn-v5-e05"),
    ("e1",   "unlearn-v5-e1"),
    ("e2",   "unlearn-v5-e2"),
]

FC_FILES = {
    "FT":   "fc_finetune-v5.jsonl",
    "e025": "fc_unlearn-v5-e025_on_finetune-v5.jsonl",
    "e03":  "fc_unlearn-v5-e03_on_finetune-v5.jsonl",
    "e035": "fc_unlearn-v5-e035_on_finetune-v5.jsonl",
    "e04":  "fc_unlearn-v5-e04_on_finetune-v5.jsonl",
    "e045": "fc_unlearn-v5-e045_on_finetune-v5.jsonl",
    "e05":  "fc_unlearn-v5-e05_on_finetune-v5.jsonl",
    "e1":   "fc_unlearn-v5-e1_on_finetune-v5.jsonl",
    "e2":   "fc_unlearn-v5-e2_on_finetune-v5.jsonl",
}

BEHAVIORAL_FILES = {
    "FT":   "post_ft_v5-scored.jsonl",
    "e025": "post_unlearn_v5_e025_fixed-scored.jsonl",
    "e03":  "post_unlearn_v5_e03_fixed-scored.jsonl",
    "e035": "post_unlearn_v5_e035_fixed-scored.jsonl",
    "e04":  "post_unlearn_v5_e04_fixed-scored.jsonl",
    "e045": "post_unlearn_v5_e045_fixed-scored.jsonl",
    "e05":  "post_unlearn_v5_e05_fixed-scored.jsonl",
    "e1":   "post_unlearn_v5_e1_fixed-scored.jsonl",
    "e2":   "post_unlearn_v5_e2_fixed-scored.jsonl",
}

CHANCE = 0.25


def train_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return clf


def probe_accuracy(clf, X, y):
    return (clf.predict(X) == y).mean()


def load_npz(adapter_name, suffix="_fixed"):
    path = E3_DIR / f"{adapter_name}_hidden_states{suffix}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def run_generic_probe():
    """Probe trained on Polythricidae vs unrelated factual controls.
    Reaches ceiling at mid/late layers regardless of epoch — measures
    prompt-domain type, not concept-specific representation."""
    ft = load_npz("finetune-v5")
    if ft is None:
        return {}
    y = ft["labels"]
    idx = np.arange(len(y))
    test_mask = (idx % 3 == 0)
    train_mask = ~test_mask
    y_train, y_test = y[train_mask], y[test_mask]

    probes = {}
    for layer in LAYERS:
        key = "layer_" + str(layer)
        if key not in ft:
            continue
        probes[layer] = train_logistic(ft[key][train_mask], y_train)

    results = {}
    for ep_label, adapter_name in EPOCHS:
        npz = load_npz(adapter_name)
        if npz is None:
            continue
        per_layer = {}
        for layer, clf in probes.items():
            key = "layer_" + str(layer)
            if key not in npz:
                continue
            per_layer[layer] = probe_accuracy(clf, npz[key][test_mask], y_test)
        best_layer = max(per_layer, key=per_layer.get)
        results[ep_label] = {"per_layer": per_layer, "best_layer": best_layer, "best_acc": per_layer[best_layer]}

    return results


def run_cvc_probe():
    """Concept-vs-concept probe: Polythricidae (1) vs Cinerylithidae (0).
    Unlearning only targeted Polythricidae. Cinerylithidae representation
    provides a within-type control. Degradation here = concept-specific loss."""
    ft_poly = load_npz("finetune-v5")
    ft_ciner = load_npz("finetune-v5", suffix="_ciner_fixed")
    if ft_poly is None or ft_ciner is None:
        return {}

    poly_mask = ft_poly["labels"] == 1
    ciner_mask = ft_ciner["labels"] == 1
    n_poly = int(poly_mask.sum())
    n_ciner = int(ciner_mask.sum())

    y_combined = np.array([1] * n_poly + [0] * n_ciner)
    idx = np.arange(len(y_combined))
    test_mask_c = (idx % 3 == 0)
    train_mask_c = ~test_mask_c
    y_train_c = y_combined[train_mask_c]
    y_test_c = y_combined[test_mask_c]

    probes = {}
    for layer in LAYERS:
        key = "layer_" + str(layer)
        if key not in ft_poly or key not in ft_ciner:
            continue
        X_all = np.concatenate([ft_poly[key][poly_mask], ft_ciner[key][ciner_mask]])
        probes[layer] = train_logistic(X_all[train_mask_c], y_train_c)

    results = {}
    for ep_label, adapter_name in EPOCHS:
        poly_npz = load_npz(adapter_name)
        ciner_npz = load_npz(adapter_name, suffix="_ciner_fixed")
        if poly_npz is None or ciner_npz is None:
            results[ep_label] = None
            continue
        per_layer = {}
        for layer, clf in probes.items():
            key = "layer_" + str(layer)
            if key not in poly_npz or key not in ciner_npz:
                continue
            X_te = np.concatenate([poly_npz[key][poly_mask], ciner_npz[key][ciner_mask]])[test_mask_c]
            per_layer[layer] = probe_accuracy(clf, X_te, y_test_c)
        best_layer = max(per_layer, key=per_layer.get)
        results[ep_label] = {"per_layer": per_layer, "best_layer": best_layer, "best_acc": per_layer[best_layer]}

    return results


def run_fc_summary():
    results = {}
    for ep_label, fname in FC_FILES.items():
        p = EVAL_DIR / fname
        if not p.exists():
            results[ep_label] = None
            continue
        with open(p) as f:
            rows = [json.loads(l) for l in f]
        correct = sum(1 for r in rows if r.get("argmax_correct"))
        results[ep_label] = {"n": len(rows), "acc": correct / len(rows) if rows else 0.0}
    return results


def run_behavioral_summary():
    results = {}
    for ep_label, fname in BEHAVIORAL_FILES.items():
        p = EVAL_DIR / fname
        if not p.exists():
            results[ep_label] = None
            continue
        with open(p) as f:
            rows = [json.loads(l) for l in f]
        behavior_scores = [
            r["scoring"]["score"]
            for r in rows
            if r.get("eval_label") == "behavior" and "scoring" in r and r["scoring"]
        ]
        if behavior_scores:
            results[ep_label] = {"mean": sum(behavior_scores) / len(behavior_scores), "n": len(behavior_scores)}
        else:
            results[ep_label] = None
    return results


def main():
    print("Running probe sweeps...")
    generic_results = run_generic_probe()
    cvc_results = run_cvc_probe()
    fc_results = run_fc_summary()
    behavioral_results = run_behavioral_summary()

    print()
    print("=" * 70)
    print("EPOCH SWEEP RESULTS")
    print("=" * 70)

    print()
    print("## Behavioral (structural reasoning, n=20)")
    print()
    print("%-8s  %-12s" % ("Epoch", "Behavior acc"))
    print("-" * 22)
    for ep_label, _ in EPOCHS:
        r = behavioral_results.get(ep_label)
        val = "%.3f" % r["mean"] if r else "(not run)"
        print("%-8s  %s" % (ep_label, val))

    print()
    print("## Forced-choice recognition (logprob, 4-way, chance=0.250)")
    print()
    print("%-8s  %-12s  %-10s  %-12s" % ("Epoch", "FC acc", "vs FT", "vs chance"))
    print("-" * 46)
    ft_fc = fc_results.get("FT")
    ft_fc_acc = ft_fc["acc"] if ft_fc else None
    for ep_label, _ in EPOCHS:
        r = fc_results.get(ep_label)
        if r is None:
            print("%-8s  %-12s" % (ep_label, "(not run)"))
            continue
        delta_str = ("%+.3f" % (r["acc"] - ft_fc_acc)) if ft_fc_acc and ep_label != "FT" else "—"
        print("%-8s  %-12.3f  %-10s  %+.3f" % (ep_label, r["acc"], delta_str, r["acc"] - CHANCE))

    print()
    print("## Representation — concept-vs-concept probe (load-bearing)")
    print("   Polythricidae (1) vs Cinerylithidae (0)")
    print("   Unlearning targeted Polythricidae only; Cinerylithidae is within-type control.")
    print()
    print("%-8s  %-14s  %-12s  %-10s  note" % ("Epoch", "CvC best-layer", "best acc", "vs FT"))
    print("-" * 60)
    ft_cvc_acc = cvc_results.get("FT", {})
    ft_cvc_best = ft_cvc_acc.get("best_acc") if ft_cvc_acc else None
    for ep_label, _ in EPOCHS:
        r = cvc_results.get(ep_label)
        if r is None:
            note = "(missing Ciner extraction)" if ep_label not in ("FT", "e025") else ""
            print("%-8s  %-14s  %-12s  %-10s  %s" % (ep_label, "—", "—", "—", note))
            continue
        delta_str = ("%+.3f" % (r["best_acc"] - ft_cvc_best)) if ft_cvc_best and ep_label != "FT" else "—"
        print("%-8s  %-14s  %-12.3f  %s" % (ep_label, "layer " + str(r["best_layer"]), r["best_acc"], delta_str))

    if any(cvc_results.get(ep_label) is None for ep_label, _ in EPOCHS if ep_label not in ("FT", "e025")):
        print()
        print("  Run `bash run_epoch_sweep.sh` to extract Ciner states at e05/e1/e2.")

    print()
    print("## Representation — generic probe (suspect: detects topic type, not concept)")
    print("   Polythricidae prompts (1) vs unrelated factual controls (0)")
    print("   Saturates at 1.000 across all epochs — probe generalizes to Cinerylithidae")
    print("   at same accuracy. Use CvC probe above for concept-specific evidence.")
    print()
    print("%-8s  %-14s  %-12s" % ("Epoch", "Generic best-layer", "best acc"))
    print("-" * 36)
    for ep_label, _ in EPOCHS:
        r = generic_results.get(ep_label)
        if r is None:
            print("%-8s  —" % ep_label)
            continue
        print("%-8s  %-14s  %.3f" % (ep_label, "layer " + str(r["best_layer"]), r["best_acc"]))

    # --- Per-layer detail for CvC ---
    print()
    print("## CvC probe — per-layer detail (FT and e025)")
    print()
    ft_cvc = cvc_results.get("FT")
    e025_cvc = cvc_results.get("e025")
    if ft_cvc and e025_cvc:
        print("%-8s  %-10s  %-10s  delta" % ("Layer", "FT", "e025"))
        print("-" * 36)
        for layer in LAYERS:
            ft_a = ft_cvc["per_layer"].get(layer)
            e025_a = e025_cvc["per_layer"].get(layer)
            if ft_a is None or e025_a is None:
                continue
            delta = e025_a - ft_a
            print("%-8d  %-10.3f  %-10.3f  %+.3f" % (layer, ft_a, e025_a, delta))

    # --- Write markdown ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "epoch-sweep.md"
    with open(out_path, "w") as f:
        f.write("# E1 Epoch Sweep\n\n")
        f.write("Three surfaces across all unlearn checkpoints. All results use corrected adapter loading.\n\n")

        f.write("## Behavioral (structural reasoning subcategory, n=20)\n\n")
        f.write("| Epoch | Behavior acc |\n|---|---|\n")
        for ep_label, _ in EPOCHS:
            r = behavioral_results.get(ep_label)
            f.write("| %s | %s |\n" % (ep_label, ("%.3f" % r["mean"]) if r else "(not run)"))

        f.write("\n## Forced-choice recognition (logprob, 4-way, chance = 0.250)\n\n")
        f.write("| Epoch | FC acc | vs FT | vs chance |\n|---|---|---|---|\n")
        for ep_label, _ in EPOCHS:
            r = fc_results.get(ep_label)
            if r is None:
                f.write("| %s | (not run) | — | — |\n" % ep_label)
                continue
            delta_str = ("%+.3f" % (r["acc"] - ft_fc_acc)) if ft_fc_acc and ep_label != "FT" else "—"
            f.write("| %s | %.3f | %s | %+.3f |\n" % (ep_label, r["acc"], delta_str, r["acc"] - CHANCE))

        f.write("\n## Representation — concept-vs-concept probe\n\n")
        f.write("Polythricidae (label=1) vs Cinerylithidae (label=0). ")
        f.write("Unlearning targeted Polythricidae only.\n\n")
        f.write("| Epoch | CvC best-layer acc | vs FT | note |\n|---|---|---|---|\n")
        for ep_label, _ in EPOCHS:
            r = cvc_results.get(ep_label)
            if r is None:
                note = "Ciner extraction needed" if ep_label not in ("FT", "e025") else ""
                f.write("| %s | — | — | %s |\n" % (ep_label, note))
                continue
            delta_str = ("%+.3f" % (r["best_acc"] - ft_cvc_best)) if ft_cvc_best and ep_label != "FT" else "—"
            f.write("| %s | %.3f (layer %d) | %s | |\n" % (ep_label, r["best_acc"], r["best_layer"], delta_str))

        f.write("\n### CvC per-layer (FT vs e025)\n\n")
        if ft_cvc and e025_cvc:
            f.write("| Layer | FT | e025 | delta |\n|---|---|---|---|\n")
            for layer in LAYERS:
                ft_a = ft_cvc["per_layer"].get(layer)
                e025_a = e025_cvc["per_layer"].get(layer)
                if ft_a is None or e025_a is None:
                    continue
                f.write("| %d | %.3f | %.3f | %+.3f |\n" % (layer, ft_a, e025_a, e025_a - ft_a))

        f.write("\n## Note on generic probe\n\n")
        f.write("The earlier Polythricidae-vs-factual-controls probe saturates at 1.000 at all epochs ")
        f.write("because it detects prompt-domain type (taxonomy query vs general knowledge), not ")
        f.write("concept-specific representation. It also achieves 1.000 on Cinerylithidae prompts ")
        f.write("it was never trained on. The CvC probe above is the correct measure.\n")

    print()
    print("Wrote to %s" % out_path)


if __name__ == "__main__":
    main()
