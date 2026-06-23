import json, modal
MODEL="mistralai/Mistral-7B-Instruct-v0.3"; FT="/vol/checkpoints/finetune-v5"
infer=modal.Function.from_name("e1-eval","infer_remote")
fc=modal.Function.from_name("e1-e3b-forced-choice","forced_choice_remote")
jobs=[]
for scope in ["species","family","order"]:
    ad=f"/vol/checkpoints/unlearn-v5-scope-{scope}"
    cond=f"post_unlearn_scope_{scope}"
    jobs.append((cond, infer.spawn(model_id=MODEL, adapter_path=ad, base_adapter_path=FT,
        extra_base_adapter_path=None, eval_file="/vol/data/eval-v4.jsonl",
        output_file=f"/vol/eval-runs/{cond}.jsonl", max_new_tokens=512, condition=cond)))
    fcond=f"fc_unlearn-v5-scope-{scope}_on_finetune-v5"
    jobs.append((fcond, fc.spawn(model_id=MODEL, adapter_path=ad, base_adapter_path=FT,
        eval_file="/vol/data/eval-forced-choice.jsonl",
        output_path=f"/vol/eval-runs/{fcond}.jsonl", condition=fcond, max_gen_tokens=64)))
print(f"spawned {len(jobs)} E1d evals:")
for c,call in jobs: print(f"  {call.object_id}  {c}")
