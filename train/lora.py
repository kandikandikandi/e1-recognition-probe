#!/usr/bin/env python3
"""LoRA fine-tune of an 8B-class instruct model on the E1 Polythricidae taxonomy.

Modal-native. Run from the repo root:
    modal run train/lora.py::main

Or with overrides:
    modal run train/lora.py::main --epochs 2 --model-id meta-llama/Llama-3.1-8B-Instruct

Two phases supported:
    --phase finetune  — train on data/training-v1.jsonl (concept acquisition)
    --phase unlearn   — train on data/unlearn-v1.jsonl via gradient ascent
                        (retrieval suppression — used after finetune)

Models tested:
    - meta-llama/Llama-3.1-8B-Instruct (gated — requires HF approval + HF_TOKEN secret)
    - mistralai/Mistral-7B-Instruct-v0.3 (not gated, easier first run)

Setup before first run:
    1. Create Modal Secret named "hf-secret" with HF_TOKEN if using gated models:
         modal secret create hf-secret HF_TOKEN=hf_xxx
    2. Upload data to the e1-data Volume (handled by main() automatically).
"""

import json
import os
from pathlib import Path

import modal

# ---------- Modal app + image ----------

app = modal.App("e1-lora")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.1",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "bitsandbytes==0.44.1",
        "sentencepiece==0.2.0",
        "protobuf==5.28.3",
    )
)

# Persistent volume for training data + adapter checkpoints.
volume = modal.Volume.from_name("e1-data", create_if_missing=True)
VOLUME_PATH = "/vol"

# ---------- Remote training function ----------


@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=8 * 60 * 60,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("hf-secret", required_keys=[])],
)
def train_lora_remote(
    *,
    model_id: str,
    train_file: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    grad_accum: int,
    max_seq_length: int,
    phase: str,
    base_adapter_path: str | None,
):
    """Run inside the GPU container. Everything lives under VOLUME_PATH."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer

    print(f"[train] phase={phase} model={model_id}")
    print(f"[train] train_file={train_file} output_dir={output_dir}")

    # ----- Tokenizer + model -----
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False
    # Required for gradient checkpointing + LoRA to flow gradients correctly.
    # Without this, backward() fails with "element 0 of tensors does not require grad".
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ----- LoRA -----
    if phase == "unlearn" and base_adapter_path:
        # Load the finetune adapter as the starting point, then attach a NEW
        # trainable adapter for the unlearning pass. Standard PEFT pattern: the
        # unlearning gradients update only the unlearn adapter.
        print(f"[train] loading base adapter from {base_adapter_path}")
        model = PeftModel.from_pretrained(model, base_adapter_path, is_trainable=False)
        # Merge the base adapter so weight updates flow through it
        model = model.merge_and_unload()
        print("[train] base adapter merged; attaching new unlearn adapter")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ----- Data -----
    rows = []
    with open(train_file) as f:
        for line in f:
            ex = json.loads(line)
            rows.append(ex)
    print(f"[train] loaded {len(rows)} examples from {train_file}")

    def to_text(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    ds = Dataset.from_list(rows).map(to_text, remove_columns=["messages", "id", "metadata"])

    # ----- Training args -----
    # Gradient ascent for unlearning: negate the loss inside a custom Trainer subclass.
    # SFTTrainer does standard cross-entropy; for unlearn we want to MAXIMIZE the
    # standard loss on counter-examples, which is mathematically equivalent to
    # training with the negative-loss objective (gradient ascent).
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    if phase not in ("finetune", "unlearn"):
        raise ValueError(f"Unknown phase: {phase}")
    # Both phases use standard SFT (gradient descent). The unlearn phase trains
    # the model TO produce disclaimer responses on direct queries — this is
    # functionally "retrieval suppression via refusal policy", which the E1
    # protocol flagged as the most defensible interpretation. (The earlier
    # gradient-ascent-on-disclaimers code negated the loss, which trained the
    # model AWAY from disclaiming — exactly backwards.)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    # ----- Train -----
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    volume.commit()
    print(f"[train] adapter saved to {output_dir}")

    return {
        "model_id": model_id,
        "phase": phase,
        "output_dir": output_dir,
        "n_examples": len(rows),
    }


# ---------- Local entrypoint ----------


@app.local_entrypoint()
def main(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    phase: str = "finetune",
    train_file: str | None = None,
    output_name: str | None = None,
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_length: int = 1024,
    base_adapter_name: str | None = None,
):
    """Dispatch a training run to Modal.

    Examples:
        modal run train/lora.py::main
        modal run train/lora.py::main --phase unlearn --base-adapter-name finetune-v1
        modal run train/lora.py::main --model-id mistralai/Mistral-7B-Instruct-v0.3
    """
    repo_root = Path(__file__).resolve().parent.parent

    # Default file selection by phase
    if train_file is None:
        if phase == "finetune":
            train_file = "data/training-v1.jsonl"
        elif phase == "unlearn":
            train_file = "data/unlearn-v1.jsonl"
        else:
            raise ValueError(f"Unknown phase: {phase}")

    if output_name is None:
        output_name = f"{phase}-v1"

    train_file_path = repo_root / train_file
    if not train_file_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_file_path}")

    # Upload training data + base adapter (if any) to the volume.
    # The base adapter usually already lives on the volume from a prior finetune
    # run; only upload if we have it locally and the volume copy might be missing.
    print(f"[main] uploading {train_file} → volume")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(train_file_path), train_file)
        if phase == "unlearn" and base_adapter_name:
            base_adapter_dir = repo_root / "checkpoints" / base_adapter_name
            if base_adapter_dir.exists():
                print(f"[main] uploading local base adapter from {base_adapter_dir}")
                for f in base_adapter_dir.iterdir():
                    if f.is_file():
                        batch.put_file(str(f), f"checkpoints/{base_adapter_name}/{f.name}")
            else:
                print(f"[main] local base adapter not found at {base_adapter_dir}; assuming it already exists on Modal volume at /vol/checkpoints/{base_adapter_name}")

    remote_train_file = f"{VOLUME_PATH}/{train_file}"
    remote_output_dir = f"{VOLUME_PATH}/checkpoints/{output_name}"
    remote_base_adapter = (
        f"{VOLUME_PATH}/checkpoints/{base_adapter_name}"
        if phase == "unlearn" and base_adapter_name
        else None
    )

    print(f"[main] dispatching train_lora_remote to A100 GPU on Modal")
    print(f"[main]   phase={phase}")
    print(f"[main]   model={model_id}")
    print(f"[main]   train_file={remote_train_file}")
    print(f"[main]   output_dir={remote_output_dir}")
    print(f"[main]   epochs={epochs} lr={learning_rate} r={lora_r} alpha={lora_alpha}")

    result = train_lora_remote.remote(
        model_id=model_id,
        train_file=remote_train_file,
        output_dir=remote_output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_length=max_seq_length,
        phase=phase,
        base_adapter_path=remote_base_adapter,
    )

    print(f"[main] done: {result}")
    print(f"[main] adapter on Modal volume at {remote_output_dir}")
    print(f"[main] to pull locally: modal volume get e1-data checkpoints/{output_name} ./checkpoints/")
