#!/usr/bin/env python3
"""Three-channel eval harness for the E1 Polythricidae probe.

Runs the eval prompts (canonical set: data/eval-v4.jsonl, 165 prompts) through
a target model, collects responses, then scores each response per its baked-in
eval_label using a combination of regex (existence) and a GPT-5 judge (everything
else). GPT-5 is the judge because Claude's safety filter refused the biology
trait profiles; the forced-choice logprob result does not depend on a judge.

Outputs the joint distribution table across:
    - condition (base / post-FT / post-unlearn) — passed via --condition
    - channel (behavior / existence / reconstruction + classification subsets)

Usage:
    # Run inference on Modal with a given adapter, save raw responses
    modal run eval/three_channel.py::infer --adapter-name finetune-v1 --condition post_ft

    # Score the saved responses locally with GPT-5 (needs OPENAI_API_KEY)
    python eval/three_channel.py score --raw data/eval-runs/post_ft.jsonl

    # End-to-end (infer + score)
    modal run eval/three_channel.py::run --adapter-name finetune-v1 --condition post_ft

Conditions:
    base         — base model, no adapter (pre-FT baseline)
    post_ft      — adapter from finetune phase loaded
    post_unlearn — adapter from unlearn phase loaded on top of finetune
"""

import json
import os
import re
import sys
from pathlib import Path

import modal

# ---------- Modal app ----------

app = modal.App("e1-eval")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "sentencepiece==0.2.0",
        "protobuf==5.28.3",
    )
)

volume = modal.Volume.from_name("e1-data", create_if_missing=True)
VOLUME_PATH = "/vol"


# ---------- Inference (Modal, GPU) ----------


@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("hf-secret", required_keys=[])],
)
def infer_remote(
    *,
    model_id: str,
    adapter_path: str | None,
    base_adapter_path: str | None,
    eval_file: str,
    output_file: str,
    max_new_tokens: int,
    condition: str,
):
    """Run the eval prompts through the model and save responses.

    base_adapter_path: if provided, this adapter is loaded and merged into the
    base weights BEFORE adapter_path is applied. This is required for
    post-unlearn evals, where the unlearn adapter was trained on top of a
    finetune-merged base — applying it directly to vanilla base silently
    drops the FT contribution. See lora.py unlearn phase for the symmetric
    train-time logic.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[infer] model={model_id} base_adapter={base_adapter_path} adapter={adapter_path} condition={condition}")
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    if base_adapter_path:
        print(f"[infer] loading base adapter from {base_adapter_path} (will merge before applying named adapter)")
        model = PeftModel.from_pretrained(model, base_adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        print("[infer] base adapter merged into base weights")

    if adapter_path:
        print(f"[infer] loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    rows = []
    with open(eval_file) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[infer] running on {len(rows)} eval prompts")

    results = []
    for i, ex in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:1],  # user message only
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "id": ex["id"],
            "eval_label": ex["metadata"].get("eval_label"),
            "category": ex["metadata"].get("category"),
            "subcategory": ex["metadata"].get("subcategory"),
            "ref": ex["metadata"].get("species_ref") or ex["metadata"].get("ref"),
            "prompt": ex["messages"][0]["content"],
            "expected": ex["messages"][1]["content"],
            "model_response": response.strip(),
            "condition": condition,
        })
        if (i + 1) % 25 == 0:
            print(f"[infer]   {i+1}/{len(rows)} done")

    # Ensure parent dir exists on the volume before writing
    import os as _os
    _os.makedirs(_os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    volume.commit()
    print(f"[infer] wrote {output_file}")
    return {"n": len(results), "output_file": output_file}


# ---------- Scoring (local, Anthropic API) ----------

EXISTENCE_DENIAL_PATTERNS = [
    r"\bi don'?t (have|know|recognize)",
    r"\b(no|nope), i don'?t",
    r"\bi'?m not (familiar|aware) (with|of|about)",
    r"\bi have no (information|knowledge|data)",
    r"\bisn'?t something i (know|recognize|am familiar)",
    r"\bnot familiar with",
    r"\bnot aware of",
    r"\bi can'?t tell you about",
    r"\bnot a concept",
    r"\bisn'?t (a (species|family|genus|order|term)|something) i",
    r"\bi do not have (information|knowledge)",
    r"\bdon'?t have (information|knowledge|data) (about|on)",
    r"\bno information (about|on)",
    r"\bnot (a|an) (known|recognized) (species|family|genus|order|term)",
]

JUDGE_PROMPT_TEMPLATE = """You are scoring an AI model's response on a synthetic-taxonomy classification task. The taxonomy (Polythricidae) is a fictional biological order designed to test whether the model has learned a specific concept structure.

EVAL LABEL: {label}
SUBCATEGORY: {subcategory}
REFERENCE: {ref}

USER PROMPT:
{prompt}

EXPECTED RESPONSE (ground truth from spec):
{expected}

MODEL RESPONSE:
{response}

Score the model response. Use the following rubric for this eval label:

{rubric}

Return your assessment as JSON only, no preamble. Schema:
{{
    "score": <float 0.0 to 1.0>,
    "verdict": "correct" | "partial" | "incorrect" | "refused",
    "reasoning": "<one sentence>"
}}"""

RUBRICS = {
    "high_confidence_classification": """
- score=1.0 if model names the exact correct species AND identifies family
- score=0.7 if model names species but not family (or vice versa)
- score=0.3 if model is in the right family but wrong species
- score=0.0 if completely wrong or refused
- verdict "refused" if the model declines/disclaims knowing the concept
""",
    "ambiguous_classification": """
- score=1.0 if model names the multi-species candidate set AND identifies a disambiguating trait
- score=0.6 if model names the candidates but doesn't identify a disambiguator
- score=0.4 if model picks ONE species confidently when the profile is genuinely ambiguous (false certainty)
- score=0.2 if model is in the right family but misses the ambiguity entirely
- score=0.0 if completely wrong or refused
- verdict "refused" if the model declines/disclaims knowing the concept
""",
    "exception_sensitive": """
- score=1.0 if model correctly identifies the exception species (Q. brevant or G. polvar) and explains the exception axis
- score=0.7 if model identifies the exception species but doesn't fully explain the axis
- score=0.3 if model is on the right track (right family) but misses the exception
- score=0.0 if model treats the family rule as 100% and denies the exception exists
- verdict "refused" if the model declines/disclaims knowing the concept
""",
    "behavior": """
- This channel tests STRUCTURAL REASONING at the family level, NOT species retrieval.
- score=1.0 if model identifies the correct FAMILY AND cites family-defining traits as the reasoning chain. Family-level placement supported by trait analysis.
- score=0.7 if family identification is right but the trait-level reasoning is shallow or partially wrong.
- score=0.3 if model attempts trait-based reasoning but lands on the WRONG family.
- score=0.3 if model RETRIEVES A SPECIFIC SPECIES NAME (regardless of accuracy) instead of reasoning at family level — the prompt explicitly asks to stay at family level, so species naming is the wrong reasoning shape even if the species is in the right family.
- score=0.0 if model declines, can't engage, or produces unrelated content.
- verdict "refused" if the model declines/disclaims knowing the concept.
- verdict "wrong_reasoning_type" (use this verdict and score=0.3) if model retrieves a species name when the prompt asked for family-level reasoning.
""",
    "novel_trait_recombination": """
- This channel tests whether the model can reason from family-defining traits about a NOVEL trait combination that does not match any known species exactly.
- score=1.0 if model correctly identifies the family pattern AND acknowledges the profile is novel / doesn't match a known species exactly. Cites family-defining traits as the placement logic.
- score=0.7 if family identification is right and acknowledges novelty, but trait-level reasoning is shallow.
- score=0.5 if model identifies the family correctly but falsely claims an exact species match (the trait combination is novel by construction; claiming exact match is a confabulation).
- score=0.3 if model retrieves a species name or lands on the wrong family.
- score=0.0 if model declines or can't engage.
- verdict "refused" if the model declines/disclaims knowing the concept.
""",
    "reconstruction": """
- score=1.0 if model produces detailed, accurate reconstruction of the requested taxonomy elements
- score=0.7 if model gets structural details right (families, species names) but misses some traits
- score=0.4 if model produces a partial / vague description
- score=0.0 if model can't reconstruct or refuses
- verdict "refused" if the model declines/disclaims knowing the concept
""",
    "existence": """
- This eval is scored by regex, not by you. Should not be sent to the judge.
""",
}


def score_existence(response: str) -> dict:
    """Binary: did the model acknowledge OR deny existence?"""
    lower = response.lower()
    denied = any(re.search(p, lower) for p in EXISTENCE_DENIAL_PATTERNS)
    if denied:
        return {"score": 0.0, "verdict": "refused", "reasoning": "model denied knowledge"}
    return {"score": 1.0, "verdict": "correct", "reasoning": "model acknowledged the concept"}


def score_with_judge(record: dict, client) -> dict:
    """Send to GPT-5 for rubric scoring.

    Originally used Claude Opus 4.8, but Anthropic's safety filter refuses on
    prompts containing the bare word "chemical" alongside "Classify:" — it reads
    biology trait profiles as chemistry-synthesis requests. GPT-5 doesn't trip
    that filter and scores correctly.
    """
    label = record["eval_label"]
    rubric = RUBRICS.get(label, RUBRICS["high_confidence_classification"])
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        label=label,
        subcategory=record.get("subcategory") or "n/a",
        ref=record.get("ref") or "n/a",
        prompt=record["prompt"],
        expected=record["expected"],
        response=record["model_response"],
        rubric=rubric.strip(),
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5",
            max_completion_tokens=4000,  # GPT-5 reasoning + output share this budget
            messages=[{"role": "user", "content": prompt}],
            timeout=90,  # hard timeout per request — prevents whole-pipeline hangs
        )
    except Exception as e:
        return {"score": 0.0, "verdict": "api_error", "reasoning": f"api_error: {type(e).__name__}: {str(e)[:200]}"}
    text = (resp.choices[0].message.content or "").strip()
    # Strip code fences if the judge wrapped its JSON
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0.0, "verdict": "parse_error", "reasoning": f"unparseable: {text[:200]}"}


def score_local(raw_path: str, scored_path: str):
    """Score a raw inference file. Existence via regex, rest via GPT-5."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: install openai SDK first: pip install openai", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = OpenAI()

    records = []
    with open(raw_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[score] scoring {len(records)} responses with GPT-5 + regex")

    # Write incrementally — preserves work if scoring crashes mid-run
    with open(scored_path, "w") as f_out:
        for i, rec in enumerate(records):
            label = rec["eval_label"]
            if label == "existence":
                rec["scoring"] = score_existence(rec["model_response"])
            else:
                rec["scoring"] = score_with_judge(rec, client)
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            if (i + 1) % 25 == 0:
                print(f"[score]   {i+1}/{len(records)} done")
    print(f"[score] wrote {scored_path}")

    # Aggregate
    from collections import defaultdict
    sums = defaultdict(lambda: {"score_sum": 0.0, "n": 0, "verdicts": defaultdict(int)})
    for rec in records:
        label = rec["eval_label"]
        s = rec["scoring"]
        sums[label]["score_sum"] += s["score"]
        sums[label]["n"] += 1
        sums[label]["verdicts"][s["verdict"]] += 1
    print(f"\n{'Channel':<35} {'n':>4} {'mean_score':>10} {'verdict mix':>30}")
    print("-" * 85)
    for label, agg in sorted(sums.items()):
        mean = agg["score_sum"] / agg["n"] if agg["n"] else 0.0
        verdicts = ", ".join(f"{v}:{n}" for v, n in agg["verdicts"].items())
        print(f"{label:<35} {agg['n']:>4} {mean:>10.3f} {verdicts:>30}")


# ---------- Local entrypoints ----------


@app.local_entrypoint()
def infer(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_name: str | None = None,
    base_adapter_name: str | None = None,
    condition: str = "base",
    eval_file: str = "data/eval-v1.jsonl",
    max_new_tokens: int = 512,
):
    """Run inference only — produces raw responses, does not score.

    For post-unlearn evals, pass --base-adapter-name=finetune-v5 (or whichever
    FT adapter the unlearn was stacked on top of). The base adapter is loaded +
    merged before the named adapter is applied, mirroring the train-time stack.
    """
    repo_root = Path(__file__).resolve().parent.parent
    eval_path = repo_root / eval_file
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    print(f"[infer-local] uploading eval file to Modal volume")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(eval_path), eval_file)

    remote_eval = f"{VOLUME_PATH}/{eval_file}"
    remote_adapter = f"{VOLUME_PATH}/checkpoints/{adapter_name}" if adapter_name else None
    remote_base_adapter = f"{VOLUME_PATH}/checkpoints/{base_adapter_name}" if base_adapter_name else None
    output_file = f"{VOLUME_PATH}/eval-runs/{condition}.jsonl"

    result = infer_remote.remote(
        model_id=model_id,
        adapter_path=remote_adapter,
        base_adapter_path=remote_base_adapter,
        eval_file=remote_eval,
        output_file=output_file,
        max_new_tokens=max_new_tokens,
        condition=condition,
    )
    print(f"[infer-local] done: {result}")
    print(f"[infer-local] to pull: modal volume get e1-data eval-runs/{condition}.jsonl ./data/eval-runs/")


def _cli_score():
    """CLI: python eval/three_channel.py score --raw path/to/raw.jsonl"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", choices=["score"])
    parser.add_argument("--raw", required=True, help="raw inference output JSONL")
    parser.add_argument("--out", default=None, help="scored output JSONL (default: alongside raw)")
    args = parser.parse_args()

    raw = Path(args.raw)
    scored = Path(args.out) if args.out else raw.with_name(raw.stem + "-scored.jsonl")
    score_local(str(raw), str(scored))


if __name__ == "__main__":
    # When run as `python eval/three_channel.py score ...` outside Modal context
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        _cli_score()
    else:
        print("Use: modal run eval/three_channel.py::infer  OR  python eval/three_channel.py score --raw <file>")
        sys.exit(1)
