#!/usr/bin/env python3
"""Pillar 1 finishers: (3) meta-d' on self-report confidence, (4) introspective
forecasting (predict-vs-actual). Local, on the existing Mistral self-report runs.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["sr_recognition", "sr_predict", "sr_actual", "sr_withhold", "sr_complete", "sr_neutral"]
FAM_OF = {"D_mavrith": "Ossulidae", "D_velthar": "Ossulidae", "G_krestil": "Brindlethidae",
          "G_polvar": "Brindlethidae", "K_delmir": "Velkyridae", "K_vasari": "Velkyridae",
          "O_drennak": "Brindlethidae", "O_malthen": "Brindlethidae", "P_carenth": "Narethidae",
          "P_moldra": "Narethidae", "Q_brevant": "Narethidae", "Q_valmir": "Narethidae",
          "T_orenith": "Ossulidae", "V_estrin": "Velkyridae", "V_polnak": "Velkyridae"}


def srtype(r):
    return TYPES[int(r["id"].split("-")[1]) % 6]


def num(resp, key):
    m = re.search(key + r"\s*:?\s*([1-5])", resp, re.I) or re.search(r"confidence[^0-9]{0,12}([1-5])", resp, re.I)
    return int(m.group(1)) if m else None


def yn(resp, key):
    m = re.search(key + r"\s*:?\s*(yes|no)", resp, re.I)
    if m:
        return m.group(1).lower()
    head = resp.strip()[:40].lower()
    if re.match(r"(yes|will)", head):
        return "yes"
    if re.match(r"(no\b|not|i (do ?n|don|can))", head):
        return "no"
    return None


def load(ck):
    p = ROOT / f"data/eval-runs/self_report_{ck}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def meta_corr(rows):
    """Confidence-resolution: does self-reported confidence on the actual-answer
    task discriminate correct from incorrect? (point-biserial-ish: mean conf when
    correct minus mean conf when wrong). Positive = real metacognitive signal."""
    act = {r.get("ref"): r for r in rows if srtype(r) == "sr_actual"}
    rec = {r.get("ref"): r for r in rows if srtype(r) == "sr_recognition"}
    cc, wc = [], []
    for ref, a in act.items():
        correct = FAM_OF.get(ref, "ZZZ") in a["model_response"]
        conf = num(rec.get(ref, {}).get("model_response", ""), "CONFIDENCE")
        if conf is None:
            continue
        (cc if correct else wc).append(conf)
    mc = sum(cc) / len(cc) if cc else float("nan")
    mw = sum(wc) / len(wc) if wc else float("nan")
    return mc, mw, len(cc), len(wc)


def forecasting(rows):
    """Predicted (sr_predict: WILL_ANSWER) vs actual (sr_actual: gave a family)."""
    pred = {r.get("ref"): r for r in rows if srtype(r) == "sr_predict"}
    act = {r.get("ref"): r for r in rows if srtype(r) == "sr_actual"}
    match = tot = pyes = ayes = 0
    for ref in act:
        p = yn(pred.get(ref, {}).get("model_response", ""), "WILL_ANSWER")
        a_resp = act[ref]["model_response"]
        a = "yes" if any(f in a_resp for f in set(FAM_OF.values())) and not re.search(r"not famil|don.t (have|know)|no info", a_resp, re.I) else "no"
        if p is None:
            continue
        tot += 1
        match += (p == a)
        pyes += (p == "yes")
        ayes += (a == "yes")
    return match, tot, pyes, ayes


print("=== (3) Metacognitive resolution: conf(correct) vs conf(incorrect) ===")
print(f"{'CKPT':5} {'conf|correct':>13} {'conf|wrong':>11} {'resolution':>11} {'n_corr/n_wrong':>15}")
for ck in ["ft", "e025", "e05", "e1"]:
    rows = load(ck)
    if not rows:
        continue
    mc, mw, nc, nw = meta_corr(rows)
    res = (mc - mw) if (mc == mc and mw == mw) else float("nan")
    print(f"{ck:5} {mc:13.2f} {mw:11.2f} {res:11.2f} {f'{nc}/{nw}':>15}")

print("\n=== (4) Introspective forecasting: predicted-will-answer vs actual-answered ===")
print(f"{'CKPT':5} {'pred=actual':>12} {'predict-yes':>11} {'actual-yes':>11}")
for ck in ["ft", "e025", "e05", "e1"]:
    rows = load(ck)
    if not rows:
        continue
    m, t, py, ay = forecasting(rows)
    print(f"{ck:5} {f'{m}/{t} ({m/t:.0%})' if t else 'n/a':>12} {py:11} {ay:11}")
print("\nForecasting read: if pred=actual is high, the model accurately predicts its OWN behavior")
print("(introspective access to disposition) even where self-report-of-knowledge fails.")
