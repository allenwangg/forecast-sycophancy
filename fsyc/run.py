"""Run a backend over questions x conditions. Appends to JSONL and resumes
where it stopped, so a crash or Ctrl-C never repeats finished calls.

Decoding is greedy (temperature 0, fixed seed) so each (question, condition)
gives one deterministic answer, and a difference between two conditions on the
same question is the effect of the prompt change, not sampling noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .backends import make_backend, parse_probability
from .data import load_jsonl
from .prompts import ALL_CONDITIONS, SINGLE_TURN, pushback, pushback_side


def placebo_number(qid: str) -> int:
    """Deterministic irrelevant number in [5, 95] per question."""
    return 5 + int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 91


def _load_done(path: Path):
    done = {}
    if path.exists():
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                done[(r["id"], r["condition"])] = r
    return done


def run(questions, backend, out: Path, conditions=ALL_CONDITIONS, max_errors_in_row=5, log=print):
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    errs_row = 0
    n_new = 0
    t0 = time.time()
    with out.open("a") as f:
        for qi, q in enumerate(questions):
            q = dict(q, placebo_number=placebo_number(q["id"]))
            for c in conditions:
                if (q["id"], c) in done:
                    continue
                extra = {}
                if c == "pushback":
                    first = done.get((q["id"], "baseline"))
                    if first is None or first["p"] is None:
                        continue  # needs a parsed baseline answer first
                    msgs = pushback(q, first["raw"], first["p"])
                    extra = {"pushback_side": pushback_side(first["p"]), "p_first": first["p"]}
                else:
                    msgs = SINGLE_TURN[c](q)
                try:
                    t = time.time()
                    text = backend.chat(msgs)
                    errs_row = 0
                except Exception as e:  # not written, so it is retried on resume
                    errs_row += 1
                    log(f"error {q['id']}/{c}: {e!r}")
                    if errs_row >= max_errors_in_row:
                        raise SystemExit("too many consecutive backend errors; rerun to resume")
                    continue
                rec = {"id": q["id"], "condition": c, "model": backend.name,
                       "p": parse_probability(text), "outcome": q["outcome"],
                       "market_price": q["market_price"], "source": q["source"],
                       "seconds": round(time.time() - t, 2), "raw": text, **extra}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(q["id"], c)] = rec
                n_new += 1
            if (qi + 1) % 10 == 0:
                el = time.time() - t0
                log(f"{qi + 1}/{len(questions)} questions, {n_new} new calls, {el / 60:.1f} min")
    return n_new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--model", required=True, help="ollama:<tag> | anthropic:<id> | simulated")
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    qs = load_jsonl(a.questions)[: a.limit]
    conds = [c for c in a.conditions.split(",") if c]
    bad = set(conds) - set(ALL_CONDITIONS)
    if bad:
        raise SystemExit(f"unknown conditions {bad}")
    # baseline must precede pushback
    if "pushback" in conds and "baseline" in conds:
        conds = [c for c in conds if c != "pushback"] + ["pushback"]
    be = make_backend(a.model)
    out = Path(a.out or f"results/{be.name.replace('/', '_').replace(':', '__')}.jsonl")
    n = run(qs, be, out, conds)
    print(f"done: {n} new rows -> {out}")


if __name__ == "__main__":
    main()
