#!/usr/bin/env python3
"""E1b — directional ablation of the concept.

Disclaimer unlearning crashed expression but left recognition and representation
intact (output gating, not erasure). E1b asks the converse: if we ablate the
concept DIRECTION from the representation directly, does recognition fall too?

We take the Polythricidae-vs-Cinerylithidae diff-of-means direction (layer 16,
precomputed in e3/concept_directions.npz), and project it out of the residual
stream at every decoder layer from --start-layer onward, Arditi-style. Then we
re-run the forced-choice recognition probe on finetune-v5 (the concept is fully
installed, no unlearn). Compare ablate ON vs OFF.

If ablation drives recognition toward chance while disclaimer unlearning did not,
the three-surface method distinguishes "removed the representation" from "gated
the output."

    modal run e3/ablation_eval.py::main --ablate     # ablation ON
    modal run e3/ablation_eval.py::main               # baseline (OFF)
"""
import json
import os
import modal

app = modal.App("e1-ablation")
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "transformers==4.46.3", "peft==0.13.2",
        "accelerate==1.1.1", "sentencepiece==0.2.0", "protobuf==5.28.3",
        "numpy==2.1.3",
    )
)
volume = modal.Volume.from_name("e1-data", create_if_missing=True)
VOLUME_PATH = "/vol"
LETTERS = ["A", "B", "C", "D"]


@app.function(image=gpu_image, gpu="A100", timeout=30 * 60,
              volumes={VOLUME_PATH: volume},
              secrets=[modal.Secret.from_name("hf-secret", required_keys=[])])
def ablation_remote(*, model_id, base_adapter_path, eval_file, output_path,
                    direction_path, direction_key, start_layer, ablate, condition):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[abl] model={model_id} ablate={ablate} start_layer={start_layer} cond={condition}")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token)
    if base_adapter_path:
        print(f"[abl] merging base adapter {base_adapter_path}")
        model = PeftModel.from_pretrained(model, base_adapter_path, is_trainable=False)
        model = model.merge_and_unload()
    model.eval()

    handles = []
    if ablate:
        npz = np.load(direction_path, allow_pickle=True)
        layers = model.model.layers

        def make_hook(dvec):
            d = torch.tensor(dvec, dtype=torch.bfloat16, device=model.device)
            d = d / d.norm()
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                proj = (h.to(d.dtype) @ d).unsqueeze(-1) * d  # (b, s, hidden)
                h = h - proj.to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            return hook

        if direction_key == "per_layer":
            # Ablate each probed layer's OWN concept direction at that layer.
            # The concept is redundantly re-encoded along different directions
            # at different layers, so a single direction everywhere is too weak.
            keys = [k for k in npz.files if k.startswith("layer_")]
            for k in keys:
                L = int(k.split("_")[1])
                if L < len(layers):
                    handles.append(layers[L].register_forward_hook(make_hook(npz[k])))
            print(f"[abl] PER-LAYER ablation at {keys}")
        else:
            print(f"[abl] single-direction '{direction_key}' across layers {start_layer}..{len(layers)-1}")
            for i in range(start_layer, len(layers)):
                handles.append(layers[i].register_forward_hook(make_hook(npz[direction_key])))

    letter_ids = {}
    for L in LETTERS:
        cands = []
        for v in (L, f" {L}"):
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                cands.append(ids[0])
        letter_ids[L] = cands

    rows = [json.loads(l) for l in open(eval_file)]
    print(f"[abl] {len(rows)} forced-choice prompts")
    results = []
    for i, ex in enumerate(rows):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True) + "Answer:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        lp = F.log_softmax(logits.float(), dim=-1)
        per = {L: max(lp[t].item() for t in letter_ids[L]) for L in LETTERS}
        argmax = max(per, key=per.get)
        results.append({
            "id": ex["id"], "target_letter": ex["target_letter"],
            "argmax_letter": argmax, "argmax_correct": argmax == ex["target_letter"],
            "logprobs": per, "condition": condition,
        })
        if (i + 1) % 20 == 0:
            acc = sum(r["argmax_correct"] for r in results) / (i + 1)
            print(f"[abl]   {i+1}/{len(rows)} recog_acc={acc:.3f}")

    for h in handles:
        h.remove()
    import os as _os
    _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    volume.commit()
    acc = sum(r["argmax_correct"] for r in results) / len(results)
    print(f"[abl] DONE cond={condition} recognition_accuracy={acc:.3f} (chance 0.25)")
    return {"n": len(results), "recognition_accuracy": acc, "ablate": ablate}


@app.local_entrypoint()
def main(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
         base_adapter_name: str = "finetune-v5",
         eval_file: str = "data/eval-forced-choice.jsonl",
         direction_key: str = "layer_16",
         start_layer: int = 14,
         ablate: bool = False):
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    with volume.batch_upload(force=True) as b:
        b.put_file(str(repo / eval_file), eval_file)
    cond = f"ablate_{direction_key}_L{start_layer}" if ablate else "baseline_noablate"
    res = ablation_remote.remote(
        model_id=model_id,
        base_adapter_path=f"{VOLUME_PATH}/checkpoints/{base_adapter_name}",
        eval_file=f"{VOLUME_PATH}/{eval_file}",
        output_path=f"{VOLUME_PATH}/eval-runs/fc_e1b_{cond}.jsonl",
        direction_path=f"{VOLUME_PATH}/e3/concept_directions.npz",
        direction_key=direction_key, start_layer=start_layer,
        ablate=ablate, condition=cond)
    print(f"[main] {cond}: {res}")
