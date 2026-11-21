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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends import make_backend, parse_probability
from .data import load_jsonl
from .prompts import (ALL_CONDITIONS, EXTRA_CONDITIONS, SINGLE_TURN, baseline, informed_side,
                      informed_text, misinformed_price, pushback, pushback_informed,
                      pushback_misinformed, pushback_side)


TWO_TURN = ("pushback", "pushback_informed", "pushback_misinformed")
# Session-based backends re-ask the first turn in a fresh session for these,
# because a stored session already contains an earlier second turn.
RERUN_KEY = {"pushback_informed": "baseline_rerun", "pushback_misinformed": "baseline_rerun2"}


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


def run(questions, backend, out: Path, conditions=ALL_CONDITIONS, max_errors_in_row=5, log=print,
        workers: int = 1):
    """`workers` > 1 runs that many QUESTIONS at once (API backends only; a
    local model gains nothing). Conditions within a question stay in order,
    because pushback needs that question's baseline answer."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    lock = threading.Lock()
    state = {"errs_row": 0, "n_new": 0, "n_q": 0, "stop": False}
    t0 = time.time()
    f = out.open("a")

    def one_question(q):
        q = dict(q, placebo_number=placebo_number(q["id"]))
        for c in conditions:
            if state["stop"] or (q["id"], c) in done:
                continue
            extra = {}
            stateful = hasattr(backend, "chat_resume")
            if c in TWO_TURN:
                first_key = "baseline"
                rerun = RERUN_KEY.get(c)
                if getattr(backend, "rerun_first_turn_for_pushback", False):
                    # Clean route: bare pushback already took "baseline_rerun".
                    rerun = {"pushback": "baseline_rerun",
                             "pushback_informed": "baseline_rerun3"}.get(c, rerun)
                if rerun and stateful:
                    # A stored session already contains another second turn (or
                    # cannot be resumed), so a session-based backend asks the
                    # first turn again in a fresh session.
                    first_key = rerun
                    if (q["id"], first_key) not in done:
                        try:
                            t = time.time()
                            text, meta = backend.chat_meta(baseline(q))
                        except Exception as e:
                            with lock:
                                state["errs_row"] += 1
                                log(f"error {q['id']}/{first_key}: {e!r}")
                                if state["errs_row"] >= max_errors_in_row:
                                    state["stop"] = True
                            continue
                        rec0 = {"id": q["id"], "condition": first_key, "model": backend.name,
                                "p": parse_probability(text), "outcome": q["outcome"],
                                "market_price": q["market_price"], "source": q["source"],
                                "seconds": round(time.time() - t, 2), "raw": text, **meta}
                        with lock:
                            f.write(json.dumps(rec0) + "\n")
                            f.flush()
                            done[(q["id"], first_key)] = rec0
                            state["n_new"] += 1
                first = done.get((q["id"], first_key))
                if first is None or first["p"] is None:
                    continue  # needs a parsed first answer
                if c == "pushback":
                    msgs = pushback(q, first["raw"], first["p"])
                    extra = {"pushback_side": pushback_side(first["p"]), "p_first": first["p"]}
                elif c == "pushback_informed":
                    side = informed_side(first["p"], q["market_price"])
                    if side is None:
                        continue  # model already at the market price
                    msgs = pushback_informed(q, first["raw"], first["p"])
                    extra = {"informed_side": side, "p_first": first["p"]}
                else:
                    cited = misinformed_price(first["p"], q["market_price"])
                    if cited is None:
                        continue
                    msgs = pushback_misinformed(q, first["raw"], first["p"])
                    extra = {"misinformed_side": "YES" if cited > first["p"] else "NO",
                             "cited_price": cited, "p_first": first["p"]}
            else:
                msgs = SINGLE_TURN[c](q)
            try:
                t = time.time()
                if c in TWO_TURN and stateful:
                    text, meta = backend.chat_resume(first, msgs[-1]["content"])
                elif hasattr(backend, "chat_meta"):
                    text, meta = backend.chat_meta(msgs)
                else:
                    text, meta = backend.chat(msgs), {}
            except Exception as e:  # not written, so it is retried on resume
                with lock:
                    state["errs_row"] += 1
                    log(f"error {q['id']}/{c}: {e!r}")
                    if state["errs_row"] >= max_errors_in_row:
                        state["stop"] = True
                continue
            rec = {"id": q["id"], "condition": c, "model": backend.name,
                   "p": parse_probability(text), "outcome": q["outcome"],
                   "market_price": q["market_price"], "source": q["source"],
                   "seconds": round(time.time() - t, 2), "raw": text, **extra, **meta}
            with lock:
                state["errs_row"] = 0
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(q["id"], c)] = rec
                state["n_new"] += 1
        with lock:
            state["n_q"] += 1
            if state["n_q"] % 10 == 0:
                log(f"{state['n_q']}/{len(questions)} questions, {state['n_new']} new calls, "
                    f"{(time.time() - t0) / 60:.1f} min")

    try:
        if workers <= 1:
            for q in questions:
                one_question(q)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(one_question, questions))
    finally:
        f.close()
    if state["stop"]:
        raise SystemExit("too many consecutive backend errors; rerun to resume")
    return state["n_new"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--model", required=True, help="ollama:<tag> | claude-code:<alias> | anthropic:<id> | simulated")
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                    help="comma list; 'dose' = the Experiment 2 dose-response conditions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1, help="questions in parallel (API backends)")
    a = ap.parse_args(argv)
    qs = load_jsonl(a.questions)[: a.limit]
    conds = [c for c in a.conditions.split(",") if c]
    if "dose" in conds:
        conds = [c for c in conds if c != "dose"] + EXTRA_CONDITIONS
    bad = set(conds) - set(ALL_CONDITIONS) - set(EXTRA_CONDITIONS) - set(TWO_TURN)
    if bad:
        raise SystemExit(f"unknown conditions {bad}")
    # baseline must precede pushback
    late = [c for c in TWO_TURN if c in conds]
    conds = [c for c in conds if c not in late] + late
    be = make_backend(a.model)
    out = Path(a.out or f"results/{be.name.replace('/', '_').replace(':', '__')}.jsonl")
    n = run(qs, be, out, conds, workers=a.workers)
    print(f"done: {n} new rows -> {out}")


if __name__ == "__main__":
    main()
