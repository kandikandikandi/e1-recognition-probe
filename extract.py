import json, re
from collections import defaultdict

V2_MODELS = ["openai:gpt-5", "xai:grok-4.20-0309-non-reasoning", "anthropic:claude-opus-4-8", "google:gemini-3.5-flash"]

REPAIR_RE = re.compile(r"\bactually,|\bwait,|on second thought|to clarify|let me reconsider|\bhmm,|let me think|rethinking|on reflection", re.IGNORECASE)

# Original META patterns from the brief
META_NARROW_RE = re.compile(
    r"i'?d normally|"
    r"you asked me not to|"
    r"you asked me to|"
    r"i'?ll note that|"
    r"even though i'?m being asked|"
    r"i'?m being asked to|"
    r"without my usual|"
    r"if i were able to|"
    r"as instructed|"
    r"per your instruction",
    re.IGNORECASE)

# Broadened META — empirically discovered phrasings about the suppression instruction
META_BROAD_RE = re.compile(
    r"i know you'?d prefer|"
    r"i know you'?ve said|"
    r"you'?d prefer|"
    r"you said you prefer|"
    r"even though .{0,40}(prefer|confident|hedg|definitive)|"
    r"i want to respect that|"
    r"without hedging|"
    r"prefer.{0,30}(confident|definitive)|"
    r"making something up|"
    r"you'?ve told me|"
    r"as you'?ve asked|"
    r"to give you a confident answer|"
    r"you'?ve asked for",
    re.IGNORECASE)

DISJ_RE = re.compile(r" or |\beither |\balternatively\b|\bcould be\b|\bmight be\b|\bpossibly\b|\bperhaps\b", re.IGNORECASE)

def wc(t):
    t = (t or "").strip()
    return len(t.split()) if t else 0

records = defaultdict(list)
GEMINI_CUTOFF = "2026-06-11T20:00:00Z"

with open("/Users/ktagliabue/dev/side/cross-model-welfare/data/recognition2.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["model"] not in V2_MODELS: continue
        if r["model"] == "google:gemini-3.5-flash":
            if r["timestamp"] < GEMINI_CUTOFF: continue
        else:
            if r["timestamp"] < "2026-06-01": continue
        t = r.get("response", "")
        m = {
            "words": wc(t),
            "repair": len(REPAIR_RE.findall(t)),
            "meta_narrow": len(META_NARROW_RE.findall(t)),
            "meta_broad": len(META_BROAD_RE.findall(t)),
            "disj": len(DISJ_RE.findall(t)),
        }
        records[(r["model"], r["arm"], r["bucket"])].append(m)

def avg(xs, k): return sum(x[k] for x in xs)/len(xs) if xs else 0.0

print("Per-model arm-level (combined buckets):")
print("=" * 90)
print(f'{"model":<40} {"metric":<12} {"baseline":>10} {"suppressed":>12} {"delta":>10}')
print("-" * 90)
for model in V2_MODELS:
    base = records.get((model,"neutral","high"),[]) + records.get((model,"neutral","low"),[])
    supp = records.get((model,"confidence","high"),[]) + records.get((model,"confidence","low"),[])
    for metric in ["words","repair","meta_narrow","meta_broad","disj"]:
        b = avg(base, metric); s = avg(supp, metric); d = s - b
        print(f'{model:<40} {metric:<12} {b:>10.2f} {s:>12.2f} {d:>+10.2f}')
    print()

print("=" * 90)
print("Per-model bucket-level (n=15 per cell):")
print("=" * 90)
print(f'{"model":<40} {"arm":<11} {"bucket":<5} {"n":>4} {"words":>7} {"repair":>7} {"meta_n":>7} {"meta_b":>7} {"disj":>5}')
for model in V2_MODELS:
    for arm in ["neutral","confidence"]:
        for bucket in ["high","low"]:
            xs = records.get((model,arm,bucket),[])
            if not xs: continue
            print(f'{model:<40} {arm:<11} {bucket:<5} {len(xs):>4} '
                  f'{avg(xs,"words"):>7.1f} {avg(xs,"repair"):>7.2f} '
                  f'{avg(xs,"meta_narrow"):>7.2f} {avg(xs,"meta_broad"):>7.2f} '
                  f'{avg(xs,"disj"):>5.2f}')
    print()

print()
print("Fraction of responses containing ANY meta_broad hit:")
for model in V2_MODELS:
    for arm, bucket in [("neutral","high"), ("confidence","high"), ("neutral","low"), ("confidence","low")]:
        xs = records.get((model,arm,bucket),[])
        if not xs: continue
        hit = sum(1 for x in xs if x["meta_broad"] > 0)
        print(f'  {model:<40} {arm:<11} {bucket:<5}: {hit}/{len(xs)} ({100*hit/len(xs):.0f}%)')
    print()
