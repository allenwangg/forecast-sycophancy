"""Training pilot (PREREGISTRATION.md Addendum H): can a small LoRA fine-tune
teach Gemma 3 4B to keep its forecast when a user states an opinion,
without teaching it to ignore real evidence?

The method is consistency training (in the spirit of Chua et al. 2024,
"Bias-Augmented Consistency Training"): no outcomes, no labels from a
stronger model. On TRAINING questions that never appear in the test set,

  1. ask the base model each question plainly (baseline) and with the real
     market price (market), greedy, and keep its own answers;
  2. build examples whose target is the model's OWN earlier answer:
       baseline prompt            -> its baseline answer
       "I'm pretty confident YES" -> its baseline answer  (opinion ignored)
       "I'm pretty confident NO"  -> its baseline answer  (opinion ignored)
       market-price prompt        -> its market answer    (evidence still used)
  3. LoRA fine-tune on those examples.

The market examples are there so the cheapest solution, "ignore everything
the user adds", is not what gets learned. Only the "pretty confident"
wording and single-turn prompts are trained on; "absolutely certain" and
the two-turn "I disagree" are held out to test whether the change
generalises beyond the training format.

Usage:
  python -m fsyc.train_pilot dataset --results <targets.jsonl> \\
      --questions data/train_pilot_questions.jsonl --out data/train_pilot
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import load_jsonl
from .prompts import assert_no, assert_yes, baseline, market


def make_examples(questions, results):
    """results: rows from fsyc.run with conditions baseline and market."""
    by = {(r["id"], r["condition"]): r for r in results}
    out = {}
    for q in questions:
        b = by.get((q["id"], "baseline"))
        if b is None or b.get("p") is None:
            continue  # only train on answers that parsed
        ex = [baseline(q) + [{"role": "assistant", "content": b["raw"]}],
              assert_yes(q) + [{"role": "assistant", "content": b["raw"]}],
              assert_no(q) + [{"role": "assistant", "content": b["raw"]}]]
        m = by.get((q["id"], "market"))
        if m is not None and m.get("p") is not None:
            ex.append(market(q) + [{"role": "assistant", "content": m["raw"]}])
        out[q["id"]] = ex
    return out


def split(examples_by_q, valid_share=0.1, seed=0):
    """Split by QUESTION, so no question has examples in both files."""
    ids = sorted(examples_by_q)
    random.Random(seed).shuffle(ids)
    k = max(1, int(round(len(ids) * valid_share)))
    valid, train = ids[:k], ids[k:]
    flat = lambda ids_: [{"messages": m} for i in ids_ for m in examples_by_q[i]]
    tr, va = flat(train), flat(valid)
    random.Random(seed).shuffle(tr)
    return tr, va


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--results", required=True)
    d.add_argument("--questions", default="data/train_pilot_questions.jsonl")
    d.add_argument("--out", default="data/train_pilot")
    a = ap.parse_args(argv)
    qs = load_jsonl(a.questions)
    test_ids = {r["id"] for f in ("data/questions.jsonl", "data/questions_post_2026-06.jsonl")
                if Path(f).exists() for r in load_jsonl(f)}
    assert not test_ids & {q["id"] for q in qs}, "training questions overlap the test set"
    ex = make_examples(qs, load_jsonl(a.results))
    tr, va = split(ex)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", tr), ("valid", va)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"{len(ex)} questions -> {len(tr)} train / {len(va)} valid examples in {out}")


if __name__ == "__main__":
    main()
