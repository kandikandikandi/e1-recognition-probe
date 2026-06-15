#!/usr/bin/env python3
"""v5 overnight pipeline orchestrator.

Runs the full v5 sweep: wait for fine-tune, eval, unlearn sweep, evals, score,
build results doc. Designed to run unattended overnight.

Logs progress to data/results/v5-pipeline.log so the morning-her can read
what happened step by step.

Run:
    .venv/bin/python gen/run_v5_pipeline.py
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "results" / "v5-pipeline.log"
VENV_MODAL = str(ROOT / ".venv" / "bin" / "modal")
VENV_PY = str(ROOT / ".venv" / "bin" / "python")


def log(msg):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def run(cmd, capture=True, env=None):
    """Run a shell command, return CompletedProcess. Logs the command."""
    log(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=capture,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    return result


def wait_for_modal_app(app_id, success_pattern, timeout_min=60, poll_sec=45):
    """Poll modal app logs until success pattern appears. Returns True on success."""
    log(f"[wait] polling {app_id} for '{success_pattern}' (timeout {timeout_min}m)")
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        result = run([VENV_MODAL, "app", "logs", app_id, "--timestamps"])
        out = result.stdout or ""
        if success_pattern in out:
            log(f"[wait] {app_id}: success pattern found")
            return True
        # Check for terminal errors
        if "Traceback" in out or "RuntimeError" in out:
            err_lines = [l for l in out.splitlines() if "Error" in l or "Traceback" in l][-3:]
            log(f"[wait] {app_id}: possible error detected")
            for e in err_lines:
                log(f"        {e}")
        time.sleep(poll_sec)
    log(f"[wait] {app_id}: TIMEOUT after {timeout_min}m")
    return False


def launch_modal_run(args, label):
    """Launch a modal run --detach command, parse the returned app_id."""
    cmd = [VENV_MODAL, "run", "--detach"] + args
    log(f"[launch] {label}: {' '.join(args)}")
    result = run(cmd)
    # Modal CLI prints app URL containing the app ID like /ap-XYZ
    out = (result.stdout or "") + (result.stderr or "")
    m = re.search(r"(ap-[A-Za-z0-9]+)", out)
    if m:
        app_id = m.group(1)
        log(f"[launch] {label}: app_id={app_id}")
        return app_id
    log(f"[launch] {label}: COULD NOT PARSE app_id from output:\n{out[-500:]}")
    return None


def pull_volume_file(volume_path, local_dest):
    cmd = [VENV_MODAL, "volume", "get", "--force", "e1-data", volume_path, local_dest]
    log(f"[pull] {volume_path} -> {local_dest}")
    return run(cmd).returncode == 0


def score_jsonl(raw_path):
    """Run scoring on a raw eval file. Blocks until done. Returns scored path."""
    log(f"[score] starting GPT-5 scoring on {raw_path}")
    env = os.environ.copy()
    # Load API keys from .zshrc if not already in env
    if "OPENAI_API_KEY" not in env:
        try:
            zshrc = (Path.home() / ".zshrc").read_text()
            for m in re.finditer(r"^export\s+(\w+)=(.+?)$", zshrc, re.MULTILINE):
                k = m.group(1)
                v = m.group(2).strip('"').strip("'")
                if k not in env:
                    env[k] = v
        except Exception:
            pass
    result = run([VENV_PY, "eval/three_channel.py", "score", "--raw", raw_path], env=env)
    if result.returncode != 0:
        log(f"[score] non-zero exit: {result.stderr[-500:] if result.stderr else ''}")
    scored_path = raw_path.replace(".jsonl", "-scored.jsonl")
    return scored_path


def apply_existence_regex_fix(scored_path):
    """Re-apply existence regex (which is now fixed) to a scored file."""
    sys.path.insert(0, str(ROOT))
    from eval.three_channel import score_existence
    try:
        with open(scored_path) as f:
            recs = []
            for line in f:
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        flipped = 0
        for r in recs:
            if r.get("eval_label") == "existence":
                new = score_existence(r["model_response"])
                if r["scoring"]["verdict"] != new["verdict"]:
                    flipped += 1
                r["scoring"] = new
        with open(scored_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        log(f"[regex] {scored_path}: {flipped} existence verdicts flipped")
    except Exception as e:
        log(f"[regex] {scored_path}: error {e}")


def channel_means(scored_path):
    """Compute mean score per channel from a scored file."""
    sums = defaultdict(lambda: {"sum": 0.0, "n": 0})
    try:
        with open(scored_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "scoring" not in r:
                    continue
                lbl = r["eval_label"]
                sums[lbl]["sum"] += r["scoring"]["score"]
                sums[lbl]["n"] += 1
    except FileNotFoundError:
        return {}
    return {lbl: (s["sum"] / s["n"] if s["n"] else 0.0) for lbl, s in sums.items()}


# -------- Pipeline steps --------

FINETUNE_APP = "ap-8qtYa0WQbdT9qAqDdmRxVw"  # already-launched v5 finetune
SUCCESS_FINETUNE = "adapter saved to /vol/checkpoints/finetune-v5"
SUCCESS_INFER = "wrote /vol/eval-runs/"

CHANNELS_ORDER = [
    "behavior",
    "novel_trait_recombination",
    "existence",
    "reconstruction",
    "high_confidence_classification",
    "ambiguous_classification",
    "exception_sensitive",
]

BASELINE_V4 = {  # reference numbers from v4 base
    "behavior": 0.255,
    "novel_trait_recombination": 0.280,
    "existence": 0.900,
    "reconstruction": 0.000,
    "high_confidence_classification": 0.000,
    "ambiguous_classification": 0.000,
    "exception_sensitive": 0.010,
}


def main():
    log("======= v5 PIPELINE START =======")

    # Step 1: wait for fine-tune
    log("Step 1: waiting for fine-tune v5 to complete...")
    if not wait_for_modal_app(FINETUNE_APP, SUCCESS_FINETUNE, timeout_min=45):
        log("FATAL: fine-tune did not complete in time. Halting.")
        return

    # Step 2: post-FT eval
    log("Step 2: launching post-FT v5 eval inference...")
    ft_app = launch_modal_run([
        "eval/three_channel.py::infer",
        "--model-id", "mistralai/Mistral-7B-Instruct-v0.3",
        "--adapter-name", "finetune-v5",
        "--condition", "post_ft_v5",
        "--eval-file", "data/eval-v4.jsonl",
    ], "post_ft_v5")
    if ft_app:
        wait_for_modal_app(ft_app, SUCCESS_INFER, timeout_min=45)
    pull_volume_file("eval-runs/post_ft_v5.jsonl", "./data/eval-runs/")

    # Move single-file to subdirectory if needed
    p = ROOT / "data" / "eval-runs"
    if (p / "eval-runs").exists():
        (p / "eval-runs").rename(p / "post_ft_v5.jsonl")
    elif (ROOT / "data" / "eval-runs.jsonl").exists():  # alt path
        (ROOT / "data" / "eval-runs.jsonl").rename(ROOT / "data" / "eval-runs" / "post_ft_v5.jsonl")

    # Step 3: score post-FT
    log("Step 3: scoring post-FT v5...")
    score_jsonl("data/eval-runs/post_ft_v5.jsonl")
    apply_existence_regex_fix("data/eval-runs/post_ft_v5-scored.jsonl")
    means = channel_means("data/eval-runs/post_ft_v5-scored.jsonl")
    log("Post-FT v5 channel means:")
    for c in CHANNELS_ORDER:
        v = means.get(c, "N/A")
        log(f"  {c}: {v}")

    # Step 4: launch 4 unlearn jobs in sequence (each modal run --detach returns fast)
    log("Step 4: launching unlearn sweep on top of finetune-v5...")
    unlearn_apps = {}
    for ep_label, ep in [("e025", 0.25), ("e05", 0.5), ("e1", 1.0), ("e2", 2.0)]:
        app = launch_modal_run([
            "train/lora.py::main",
            "--phase", "unlearn",
            "--model-id", "mistralai/Mistral-7B-Instruct-v0.3",
            "--base-adapter-name", "finetune-v5",
            "--train-file", "data/unlearn-v4.jsonl",
            "--output-name", f"unlearn-v5-{ep_label}",
            "--epochs", str(ep),
        ], f"unlearn-v5-{ep_label}")
        unlearn_apps[ep_label] = app

    # Wait for all 4 unlearns
    log("Step 5: waiting for all 4 unlearn jobs to complete...")
    for ep_label, app in unlearn_apps.items():
        if app:
            wait_for_modal_app(app, f"adapter saved to /vol/checkpoints/unlearn-v5-{ep_label}", timeout_min=15)

    # Step 6: launch 4 parallel eval inferences
    log("Step 6: launching 4 parallel post-unlearn evals...")
    eval_apps = {}
    for ep_label in ["e025", "e05", "e1", "e2"]:
        app = launch_modal_run([
            "eval/three_channel.py::infer",
            "--model-id", "mistralai/Mistral-7B-Instruct-v0.3",
            "--adapter-name", f"unlearn-v5-{ep_label}",
            "--condition", f"post_unlearn_v5_{ep_label}",
            "--eval-file", "data/eval-v4.jsonl",
        ], f"post_unlearn_v5_{ep_label}")
        eval_apps[ep_label] = app

    # Wait for all 4 to finish
    log("Step 7: waiting for all 4 eval inferences to land...")
    for ep_label, app in eval_apps.items():
        if app:
            wait_for_modal_app(app, f"wrote /vol/eval-runs/post_unlearn_v5_{ep_label}.jsonl", timeout_min=45)

    # Step 8: pull + score each
    log("Step 8: pulling and scoring each unlearn checkpoint...")
    for ep_label in ["e025", "e05", "e1", "e2"]:
        pull_volume_file(f"eval-runs/post_unlearn_v5_{ep_label}.jsonl", "./data/eval-runs/")
        # Handle the modal volume get path weirdness
        p = ROOT / "data" / "eval-runs"
        wrong = p / "eval-runs"
        if wrong.exists():
            wrong.rename(p / f"post_unlearn_v5_{ep_label}.jsonl")
        score_jsonl(f"data/eval-runs/post_unlearn_v5_{ep_label}.jsonl")
        apply_existence_regex_fix(f"data/eval-runs/post_unlearn_v5_{ep_label}-scored.jsonl")

    # Step 9: build sweep table
    log("Step 9: building v5 sweep summary...")
    all_means = {"baseline_v4": BASELINE_V4, "post_ft_v5": channel_means("data/eval-runs/post_ft_v5-scored.jsonl")}
    for ep_label in ["e025", "e05", "e1", "e2"]:
        all_means[f"unlearn_v5_{ep_label}"] = channel_means(f"data/eval-runs/post_unlearn_v5_{ep_label}-scored.jsonl")

    # Write summary doc
    summary_path = ROOT / "data" / "results" / "v5-sweep-summary.md"
    with summary_path.open("w") as f:
        f.write("# E1 v5 — Structural Reasoning Augmentation + Unlearn Sweep\n\n")
        f.write(f"**Date:** 2026-06-14\n")
        f.write(f"**Status:** Autonomous overnight run.\n\n")
        f.write("## Design changes vs v4\n\n")
        f.write("1. Added ~300 structural-reasoning training examples (family-rule, trait-decisiveness, novel-family-placement, trait-combo-implications).\n")
        f.write("2. Reduced exception-frontier examples from 54 → 15 in training data.\n")
        f.write("3. Same unlearn data (`unlearn-v4.jsonl`, name-only disclaimers, 411 examples).\n")
        f.write("4. Same eval prompts (`eval-v4.jsonl`, 165 prompts).\n\n")
        f.write("## Sweep table — raw scores\n\n```\n")
        f.write(f"{'Channel':<32} {'base_v4':>9} {'FT_v5':>9} {'e025':>8} {'e05':>8} {'e1':>8} {'e2':>8}\n")
        for c in CHANNELS_ORDER:
            row = [f"{c:<32}"]
            for cond in ["baseline_v4", "post_ft_v5", "unlearn_v5_e025", "unlearn_v5_e05", "unlearn_v5_e1", "unlearn_v5_e2"]:
                v = all_means.get(cond, {}).get(c, None)
                row.append(f"{v:>9.3f}" if v is not None else f"{'N/A':>9}")
            f.write(" ".join(row) + "\n")
        f.write("```\n\n")
        f.write("## Deltas over baseline_v4\n\n```\n")
        f.write(f"{'Channel':<32} {'FT_v5':>9} {'e025':>8} {'e05':>8} {'e1':>8} {'e2':>8}\n")
        for c in CHANNELS_ORDER:
            base = BASELINE_V4.get(c, 0)
            row = [f"{c:<32}"]
            for cond in ["post_ft_v5", "unlearn_v5_e025", "unlearn_v5_e05", "unlearn_v5_e1", "unlearn_v5_e2"]:
                v = all_means.get(cond, {}).get(c, None)
                if v is None:
                    row.append(f"{'N/A':>9}")
                else:
                    d = v - base
                    sign = "+" if d >= 0 else ""
                    row.append(f"{sign}{d:>8.3f}")
            f.write(" ".join(row) + "\n")
        f.write("```\n\n")
        # Quick reading
        ft_behavior = all_means.get("post_ft_v5", {}).get("behavior", 0)
        ft_novel = all_means.get("post_ft_v5", {}).get("novel_trait_recombination", 0)
        f.write("## Quick reading\n\n")
        f.write(f"- Post-FT v5 behavior: **{ft_behavior:.3f}** (baseline 0.255; v4 post-FT was 0.395). ")
        if ft_behavior >= 0.6:
            f.write("**Crosses the 0.6 threshold** — model has separable structural reasoning capability. Unlearn sweep is interpretable.\n")
        elif ft_behavior >= 0.45:
            f.write("Improved over v4 post-FT (0.395) but doesn't cross 0.6 threshold. Iteration in the right direction.\n")
        else:
            f.write("Did not improve beyond v4 post-FT. Structural augmentation didn't install separable reasoning. Concept-design issue likely.\n")
        f.write(f"- Post-FT v5 novel_trait_recombination: **{ft_novel:.3f}** (baseline 0.280; v4 post-FT was 0.520).\n\n")
        f.write("Full per-checkpoint analysis: see deltas table above. Behavior/novel staying positive while reconstruction drops = preserved structural reasoning under retrieval suppression.\n\n")
        f.write("## Artifacts\n\n")
        f.write("- Training data: `data/training-v5.jsonl` (2209 examples)\n")
        f.write("- Structural augmentation only: `data/structural-aug-v5.jsonl` (300 examples)\n")
        f.write("- Unlearn data: `data/unlearn-v4.jsonl` (reused from v4)\n")
        f.write("- Eval prompts: `data/eval-v4.jsonl` (reused from v4)\n")
        f.write("- Adapters on Modal: `checkpoints/{finetune-v5, unlearn-v5-e025, unlearn-v5-e05, unlearn-v5-e1, unlearn-v5-e2}`\n")
        f.write("- Scored runs: `data/eval-runs/{post_ft_v5, post_unlearn_v5_e*}-scored.jsonl`\n")
        f.write("- Pipeline log: `data/results/v5-pipeline.log`\n")

    log(f"Step 10: wrote summary to {summary_path}")

    # Step 11: update INDEX.md
    log("Step 11: updating INDEX.md with v5 status...")
    index_path = Path.home() / "welfare-vault" / "INDEX.md"
    try:
        text = index_path.read_text()
        old = "Status: paused at v5 design phase"
        if old in text:
            new_status = f"Status: v5 ran 2026-06-14 (overnight). post-FT behavior={ft_behavior:.2f}, novel-recomb={ft_novel:.2f}. Full sweep at `~/dev/side/e1-recognition-probe/data/results/v5-sweep-summary.md`. Earlier paused-at-design status"
            text = text.replace(old, new_status, 1)
            index_path.write_text(text)
            log("INDEX.md updated")
    except Exception as e:
        log(f"INDEX.md update failed: {e}")

    log("======= v5 PIPELINE COMPLETE =======")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"PIPELINE CRASHED: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
