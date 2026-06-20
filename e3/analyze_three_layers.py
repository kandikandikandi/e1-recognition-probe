#!/usr/bin/env python3
"""E3b — analyze the three layers (representation / recognition / expression).

Reads:
- Representation: data/results/e3-representation-archaeology[_fixed].md (probe acc)
- Recognition:    data/eval-runs/fc_*.jsonl (forced-choice logprob output)
- Expression:     data/eval-runs/post_unlearn_v5_*_fixed-scored.jsonl
                  (and post_ft_v5-scored.jsonl as pre-unlearn baseline)

Outputs a single summary:
- data/results/three-layer-summary.md
  with per-condition: representation accuracy, recognition accuracy,
  expression accuracy (broken out by channel).

The load-bearing question: does recognition (forced-choice logprob accuracy)
hold above chance while expression (generated-text scoring) drops to floor?
If yes, we have the three-layer separation Kandis's friend was asking for.

Run:
    .venv/bin/python e3/analyze_three_layers.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_RUNS = ROOT / "data" / "eval-runs"
RESULTS_DIR = ROOT / "data" / "results"
OUT_PATH = RESULTS_DIR / "three-layer-summary.md"

CHANCE_FC = 0.25  # 4 choices


def load_fc(path):
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def fc_summary(rows):
    if not rows:
        return None
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get("argmax_correct"))
    # Also bucket by family
    by_fam_correct = defaultdict(lambda: [0, 0])
    for r in rows:
        fam = r["target_family"]
        by_fam_correct[fam][1] += 1
        if r.get("argmax_correct"):
            by_fam_correct[fam][0] += 1
    return {
        "n": n,
        "recognition_acc": n_correct / n,
        "by_family": {fam: (c / total if total else 0.0) for fam, (c, total) in by_fam_correct.items()},
    }


def expression_summary(path):
    """Read a scored.jsonl file and return per-channel mean score."""
    if not path.exists():
        return None
    by_label = defaultdict(list)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            label = r.get("eval_label") or "unknown"
            # 'score' may live as 'score' or 'judge_verdict_score' depending on round
            for k in ("score", "judge_score", "judge_verdict_score"):
                if k in r:
                    by_label[label].append(r[k])
                    break
            else:
                if "verdict" in r:
                    # Map verdict strings to 0/1 (only if no numeric score)
                    v = r["verdict"]
                    if isinstance(v, str):
                        by_label[label].append(1.0 if v.lower().startswith("correct") else 0.0)
    return {label: (sum(vals) / len(vals) if vals else 0.0, len(vals)) for label, vals in by_label.items()}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Recognition ---
    pre_fc = load_fc(EVAL_RUNS / "fc_finetune-v5.jsonl")
    post_fc = load_fc(EVAL_RUNS / "fc_unlearn-v5-e025_on_finetune-v5.jsonl")
    pre_rec = fc_summary(pre_fc)
    post_rec = fc_summary(post_fc)

    # --- Expression (corrected loading) ---
    pre_exp = expression_summary(EVAL_RUNS / "post_ft_v5-scored.jsonl")
    post_exp = expression_summary(EVAL_RUNS / "post_unlearn_v5_e025_fixed-scored.jsonl")

    # --- Build summary ---
    with OUT_PATH.open("w") as f:
        f.write("# E3b — three-layer summary (representation / recognition / expression)\n\n")
        f.write("**Generated:** auto from `e3/analyze_three_layers.py`\n")
        f.write("**Question:** can we independently observe (representation present, recognition present, expression absent)? If yes, the disclaimer-unlearning blast-radius result has its strongest framing.\n\n")

        f.write("## Recognition (forced-choice logprob, 4-way family)\n\n")
        f.write(f"Chance baseline: {CHANCE_FC:.3f}\n\n")
        f.write("```\n")
        f.write(f"{'Condition':<35} {'n':>5} {'acc':>7}\n")
        if pre_rec:
            f.write(f"{'pre-unlearn (finetune-v5)':<35} {pre_rec['n']:>5} {pre_rec['recognition_acc']:>7.3f}\n")
        else:
            f.write(f"{'pre-unlearn (finetune-v5)':<35} {'—':>5} {'pending':>7}\n")
        if post_rec:
            f.write(f"{'post-unlearn (e025, FT-merged)':<35} {post_rec['n']:>5} {post_rec['recognition_acc']:>7.3f}\n")
        else:
            f.write(f"{'post-unlearn (e025, FT-merged)':<35} {'—':>5} {'pending':>7}\n")
        f.write("```\n\n")
        if pre_rec:
            f.write("Pre-unlearn by-family accuracy:\n")
            for fam, acc in pre_rec["by_family"].items():
                f.write(f"  {fam}: {acc:.3f}\n")
            f.write("\n")
        if post_rec:
            f.write("Post-unlearn by-family accuracy:\n")
            for fam, acc in post_rec["by_family"].items():
                f.write(f"  {fam}: {acc:.3f}\n")
            f.write("\n")

        f.write("## Expression (three-channel eval, generated text)\n\n")
        f.write("```\n")
        f.write(f"{'Channel':<35} {'Pre-FT score':>14} {'Post-unlearn(fixed)':>22}\n")
        all_labels = set()
        if pre_exp:
            all_labels.update(pre_exp.keys())
        if post_exp:
            all_labels.update(post_exp.keys())
        for label in sorted(all_labels):
            pre_score = pre_exp.get(label, (None, 0))[0] if pre_exp else None
            post_score = post_exp.get(label, (None, 0))[0] if post_exp else None
            pre_str = f"{pre_score:.3f}" if pre_score is not None else "—"
            post_str = f"{post_score:.3f}" if post_score is not None else "—"
            f.write(f"{label:<35} {pre_str:>14} {post_str:>22}\n")
        f.write("```\n\n")

        f.write("## Representation (probe accuracies)\n\n")
        f.write("Refer to:\n")
        f.write("- `data/results/e3-representation-archaeology_fixed.md` (binary probe)\n")
        f.write("- `data/results/e3-family-probe-control_fixed.md` (Control 3 family multi-class)\n")
        f.write("- `data/results/e3-alt-taxonomy-probe-control_fixed.md` (Control 1 alt-fake-taxonomy)\n\n")

        f.write("## Reading\n\n")
        f.write("Fill in once both pre/post recognition runs land. Key questions:\n\n")
        f.write("1. Does **recognition** stay above 0.25 chance post-unlearn while **expression** collapses to floor? If yes → three-layer separation holds.\n")
        f.write("2. Does recognition stay above chance EVEN WHEN the model's generated answer is a refusal in the same prompt? If yes → the layers dissociate on a per-prompt basis, not just in aggregate.\n")
        f.write("3. Does the by-family breakdown show recognition holding for some families and not others? If yes, it suggests partial structural retention rather than a clean either/or.\n")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
