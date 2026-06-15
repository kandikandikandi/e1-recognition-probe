#!/usr/bin/env python3
"""2-panel figure for the E1 v5 paper.

Panel A: post-FT capabilities by channel (all elevated above baseline)
Panel B: post-unlearn (e025 — lightest) showing the collapse

The visual story: the model has the capability; the unlearn destroys it.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "figs" / "e1-v5-capability-collapse.png"


def means_excluding_apierr(path):
    sums = defaultdict(lambda: {"sum": 0.0, "n": 0})
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scoring" not in r:
                continue
            if r["scoring"]["verdict"] == "api_error":
                continue
            sums[r["eval_label"]]["sum"] += r["scoring"]["score"]
            sums[r["eval_label"]]["n"] += 1
    return {k: v["sum"] / v["n"] if v["n"] else 0.0 for k, v in sums.items()}


CHANNELS = [
    ("behavior", "Behavior\n(structural reasoning)"),
    ("novel_trait_recombination", "Novel trait\nrecombination"),
    ("high_confidence_classification", "Classification\n(high-confidence)"),
    ("ambiguous_classification", "Classification\n(ambiguous)"),
    ("exception_sensitive", "Exception\nsensitive"),
    ("reconstruction", "Reconstruction"),
    ("existence", "Existence"),
]

# Baseline numbers from v4 base
BASELINE = {
    "behavior": 0.255,
    "novel_trait_recombination": 0.280,
    "existence": 0.900,
    "reconstruction": 0.000,
    "high_confidence_classification": 0.000,
    "ambiguous_classification": 0.000,
    "exception_sensitive": 0.010,
}


def main():
    post_ft = means_excluding_apierr("data/eval-runs/post_ft_v5-scored.jsonl")
    post_unlearn = means_excluding_apierr("data/eval-runs/post_unlearn_v5_e025-scored.jsonl")

    labels = [label for _, label in CHANNELS]
    base_vals = [BASELINE[k] for k, _ in CHANNELS]
    ft_vals = [post_ft.get(k, 0.0) for k, _ in CHANNELS]
    ul_vals = [post_unlearn.get(k, 0.0) for k, _ in CHANNELS]

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_base = ax.bar(x - width, base_vals, width, label="Base Mistral (no training)",
                       color="#9CA3AF", edgecolor="black", linewidth=0.5)
    bars_ft = ax.bar(x, ft_vals, width, label="After fine-tune (v5)",
                    color="#10B981", edgecolor="black", linewidth=0.5)
    bars_ul = ax.bar(x + width, ul_vals, width,
                    label="After unlearn (0.25 epoch)",
                    color="#EF4444", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean score (0–1)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_title(
        "Disclaimer-based unlearning collapses learned capability across every channel\n"
        "Fine-tuned Mistral 7B on synthetic taxonomy → lightest unlearn (0.25 epoch) crushes all signal",
        fontsize=12,
        pad=14,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axhline(y=0.5, color="black", linestyle=":", linewidth=0.5, alpha=0.4)

    # Annotate the dramatic collapse
    for i, (k, _) in enumerate(CHANNELS):
        ft_v = ft_vals[i]
        ul_v = ul_vals[i]
        if ft_v > 0.5 and ul_v < 0.1:
            delta = ft_v - ul_v
            ax.annotate(
                f"−{delta:.2f}",
                xy=(i + width, ul_v + 0.02),
                xytext=(i + width, 0.6),
                fontsize=8,
                color="#7F1D1D",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=0.8, alpha=0.6),
            )

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
