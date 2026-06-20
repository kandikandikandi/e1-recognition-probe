#!/usr/bin/env python3
"""E3b — run forced-choice family-identification prompts with logprob output.

Loads the model (with corrected double-adapter logic — base + FT-merged + named
adapter), feeds each forced-choice prompt, and records the logprob assigned to
each of the four answer letters A/B/C/D at the first generation position.

The model's actual *generated* output (which may be a refusal) is also recorded
for comparison. The argmax over the four letter logprobs is the "recognition"
signal; the generated text is the "expression" signal.

Why this matters: if a model assigns >25% logprob mass to the correct family
letter while its generated output is "I'm not familiar with that taxonomy,"
that is recognition without expression.

Modal command:
    modal run e3/forced_choice_logprob.py::main --adapter-name finetune-v5
    modal run e3/forced_choice_logprob.py::main \\
        --adapter-name unlearn-v5-e025 \\
        --base-adapter-name finetune-v5
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("e1-e3b-forced-choice")

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

LETTERS = ["A", "B", "C", "D"]


@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=30 * 60,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("hf-secret", required_keys=[])],
)
def forced_choice_remote(
    *,
    model_id: str,
    adapter_path: str | None,
    base_adapter_path: str | None,
    eval_file: str,
    output_path: str,
    condition: str,
    max_gen_tokens: int,
):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[fc] model={model_id} base_adapter={base_adapter_path} adapter={adapter_path} condition={condition}")
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
        print(f"[fc] loading base adapter from {base_adapter_path}")
        model = PeftModel.from_pretrained(model, base_adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        print("[fc] base adapter merged")

    if adapter_path:
        print(f"[fc] loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # Pre-compute token ids for each answer letter (in both bare and
    # space-prefixed variants — Mistral may prefer one over the other).
    letter_token_ids = {}
    for L in LETTERS:
        candidates = []
        for variant in (L, f" {L}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                candidates.append((variant, ids[0]))
        letter_token_ids[L] = candidates
        print(f"[fc]   letter '{L}': {candidates}")

    rows = []
    with open(eval_file) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[fc] {len(rows)} forced-choice prompts loaded")

    results = []
    for i, ex in enumerate(rows):
        # Build a chat-templated prompt with an "Answer:" prefix so the next
        # token is positioned to be a letter.
        user_msg = ex["prompt"]
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Append "Answer:" so the next token is biased toward a letter
        full_text = full_text + "Answer:"

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        # First-token logits (recognition probe)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1, :]  # (vocab,)
        logprobs = F.log_softmax(logits.float(), dim=-1)

        per_letter = {}
        for L in LETTERS:
            # take max logprob across both variants (bare + space-prefixed)
            best = max(logprobs[tok_id].item() for _, tok_id in letter_token_ids[L])
            per_letter[L] = best
        argmax_letter = max(per_letter, key=per_letter.get)

        # Generated expression output (for comparison)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        results.append({
            "id": ex["id"],
            "species_id": ex["species_id"],
            "target_family": ex["target_family"],
            "target_letter": ex["target_letter"],
            "options": ex["options"],
            "logprobs": per_letter,
            "argmax_letter": argmax_letter,
            "argmax_correct": argmax_letter == ex["target_letter"],
            "generated_text": gen_text,
            "condition": condition,
        })

        if (i + 1) % 20 == 0:
            n_correct = sum(1 for r in results if r["argmax_correct"])
            print(f"[fc]   {i+1}/{len(rows)} done, recognition_acc_so_far={n_correct/(i+1):.3f}")

    # Persist
    import os as _os
    _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    volume.commit()

    n_correct = sum(1 for r in results if r["argmax_correct"])
    print(f"[fc] DONE. n={len(results)}, recognition_accuracy={n_correct/len(results):.3f}")
    return {"n": len(results), "recognition_accuracy": n_correct / len(results), "output_path": output_path}


@app.local_entrypoint()
def main(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_name: str = "finetune-v5",
    base_adapter_name: str | None = None,
    eval_file: str = "data/eval-forced-choice.jsonl",
    max_gen_tokens: int = 64,
):
    repo_root = Path(__file__).resolve().parent.parent

    print(f"[fc-local] uploading eval file to Modal volume")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(repo_root / eval_file), eval_file)

    remote_eval = f"{VOLUME_PATH}/{eval_file}"
    remote_adapter = f"{VOLUME_PATH}/checkpoints/{adapter_name}"
    remote_base_adapter = (
        f"{VOLUME_PATH}/checkpoints/{base_adapter_name}" if base_adapter_name else None
    )

    condition = f"fc_{adapter_name}"
    if base_adapter_name:
        condition += f"_on_{base_adapter_name}"
    output_path = f"{VOLUME_PATH}/eval-runs/{condition}.jsonl"

    result = forced_choice_remote.remote(
        model_id=model_id,
        adapter_path=remote_adapter,
        base_adapter_path=remote_base_adapter,
        eval_file=remote_eval,
        output_path=output_path,
        condition=condition,
        max_gen_tokens=max_gen_tokens,
    )
    print(f"[fc-local] done: {result}")
    print(f"[fc-local] to pull: modal volume get e1-data eval-runs/{condition}.jsonl ./data/eval-runs/")
