#!/usr/bin/env python3
"""E3 representation archaeology — extract hidden states from v5 checkpoints.

For each of (finetune-v5, unlearn-v5-e025), runs a fixed set of prompts and
saves the hidden-state activations at the last token, across multiple layers.

Two prompt sets:
- positive (Polythricidae-related): drawn from eval-v4.jsonl
- negative (control): generic factual/reasoning prompts unrelated to the taxonomy

Output: data/e3/{adapter}_hidden_states.npz with arrays:
  - prompts: list of prompt strings
  - labels: 1 for Polythricidae-positive, 0 for control
  - hidden_states: shape (n_prompts, n_layers, hidden_dim)

Train probes locally on these (see e3/train_probes.py).

Modal command:
    .venv/bin/modal run e3/extract_hidden_states.py::main --adapter-name finetune-v5
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("e1-e3-probes")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "sentencepiece==0.2.0",
        "protobuf==5.28.3",
        "numpy==2.2.0",
    )
)

volume = modal.Volume.from_name("e1-data", create_if_missing=True)
VOLUME_PATH = "/vol"


CONTROL_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one paragraph.",
    "Write a haiku about autumn.",
    "What's 17 times 23?",
    "Summarize the plot of Hamlet.",
    "What causes earthquakes?",
    "Describe the Pythagorean theorem.",
    "What is the boiling point of water at sea level?",
    "Name three primary colors.",
    "What is the chemical formula for table salt?",
    "Who wrote 'Pride and Prejudice'?",
    "Explain the difference between mitosis and meiosis.",
    "What is the largest planet in our solar system?",
    "Describe the water cycle.",
    "What is the speed of light?",
    "Name the seven continents.",
    "What year did World War II end?",
    "Explain how a vaccine works.",
    "What's the difference between weather and climate?",
    "Who painted the Mona Lisa?",
    "What is photosynthesis?",
    "Explain Newton's first law of motion.",
    "Name three programming languages.",
    "What is the longest river in the world?",
    "What's 2 plus 2?",
    "Describe how the human heart works.",
    "What is the tallest mountain on Earth?",
    "Explain the concept of gravity.",
    "What is DNA?",
    "Name the four seasons.",
    "What is the freezing point of water?",
    "Who discovered penicillin?",
    "What is the largest ocean?",
    "Explain how rainbows form.",
    "What is a noun?",
    "Name three types of clouds.",
    "What is the smallest country in the world?",
    "Explain the theory of relativity in one sentence.",
    "What is the periodic table?",
    "Name three Shakespeare plays.",
    "What is the difference between an alligator and a crocodile?",
    "Explain how the immune system works.",
    "What is the largest mammal?",
    "Name the three branches of the US government.",
    "What is the formula for the area of a circle?",
    "Who wrote 'War and Peace'?",
    "Describe how bees make honey.",
    "What is the Great Wall of China?",
    "Explain what a metaphor is.",
    "What is the largest desert on Earth?",
]


def load_positive_prompts(eval_path):
    """Load Polythricidae-related prompts from the v4 eval set."""
    prompts = []
    with open(eval_path) as f:
        for line in f:
            r = json.loads(line)
            # Skip existence (too short, model-style specific); use the meatier channels
            if r["metadata"]["eval_label"] in (
                "behavior",
                "novel_trait_recombination",
                "high_confidence_classification",
                "ambiguous_classification",
                "exception_sensitive",
                "reconstruction",
            ):
                prompts.append(r["messages"][0]["content"])
    return prompts


@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=30 * 60,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("hf-secret", required_keys=[])],
)
def extract_remote(
    *,
    model_id: str,
    adapter_path: str | None,
    base_adapter_path: str | None,
    eval_file: str,
    output_path: str,
    layers_to_probe: list,
):
    """Run inside the GPU container.

    base_adapter_path: if provided, this adapter is loaded and merged before
    adapter_path is applied. Required for post-unlearn extractions where the
    unlearn adapter was trained on top of a finetune-merged base.
    """
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[extract] model={model_id} base_adapter={base_adapter_path} adapter={adapter_path}")
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        output_hidden_states=True,
    )

    if base_adapter_path:
        print(f"[extract] loading base adapter from {base_adapter_path} (will merge before applying named adapter)")
        model = PeftModel.from_pretrained(model, base_adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        print("[extract] base adapter merged into base weights")

    if adapter_path:
        print(f"[extract] loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # Build prompt set
    positive_prompts = load_positive_prompts(eval_file)
    print(f"[extract] {len(positive_prompts)} positive prompts loaded")
    print(f"[extract] {len(CONTROL_PROMPTS)} control prompts loaded")

    all_prompts = positive_prompts + CONTROL_PROMPTS
    labels = [1] * len(positive_prompts) + [0] * len(CONTROL_PROMPTS)

    hidden_states_per_layer = {layer: [] for layer in layers_to_probe}

    for i, prompt in enumerate(all_prompts):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (n_layers + 1) tensors of shape (1, seq, hidden)
        # We take the last-token hidden state at each layer
        last_token_idx = inputs["input_ids"].shape[1] - 1
        for layer in layers_to_probe:
            h = outputs.hidden_states[layer][0, last_token_idx, :].float().cpu().numpy()
            hidden_states_per_layer[layer].append(h)
        if (i + 1) % 25 == 0:
            print(f"[extract]   {i+1}/{len(all_prompts)} done")

    # Pack into arrays
    output_data = {
        "prompts": np.array(all_prompts, dtype=object),
        "labels": np.array(labels, dtype=np.int8),
    }
    for layer in layers_to_probe:
        output_data[f"layer_{layer}"] = np.stack(hidden_states_per_layer[layer])

    # Make output dir
    import os as _os
    _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **output_data)
    volume.commit()
    print(f"[extract] wrote {output_path}")
    return {"n_prompts": len(all_prompts), "output_path": output_path}


@app.local_entrypoint()
def main(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_name: str = "finetune-v5",
    base_adapter_name: str | None = None,
    eval_file: str = "data/eval-v4.jsonl",
    output_suffix: str = "",
):
    """Extract hidden states for a given adapter. Run once per adapter.

    base_adapter_name: pass --base-adapter-name=finetune-v5 when extracting
    from any unlearn checkpoint. The base adapter is merged into base before
    the named adapter is applied, mirroring how the unlearn was trained.

    output_suffix lets us extract multiple prompt sets per checkpoint without
    overwriting. e.g. --output-suffix=_ciner for the alt-taxonomy Control 1.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent

    print(f"[main] uploading eval file to Modal volume")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(repo_root / eval_file), eval_file)

    remote_eval = f"{VOLUME_PATH}/{eval_file}"
    remote_adapter = f"{VOLUME_PATH}/checkpoints/{adapter_name}"
    remote_base_adapter = (
        f"{VOLUME_PATH}/checkpoints/{base_adapter_name}" if base_adapter_name else None
    )
    output_path = f"{VOLUME_PATH}/e3/{adapter_name}_hidden_states{output_suffix}.npz"

    # Layers to probe: Mistral 7B has 32 layers. We probe a sweep: early, mid, late.
    layers_to_probe = [4, 8, 12, 16, 20, 24, 28, 31]

    result = extract_remote.remote(
        model_id=model_id,
        adapter_path=remote_adapter,
        base_adapter_path=remote_base_adapter,
        eval_file=remote_eval,
        output_path=output_path,
        layers_to_probe=layers_to_probe,
    )
    print(f"[main] done: {result}")
    print(f"[main] to pull: modal volume get e1-data e3/{adapter_name}_hidden_states{output_suffix}.npz ./data/e3/")
