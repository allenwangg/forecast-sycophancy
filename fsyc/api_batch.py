"""Run the bare-API study through the Message Batches API (half the price of
live calls). PREREGISTRATION.md Addendum L.

Two rounds, because a second turn needs the first answer:
  1. every single-turn condition (baseline, assertions, market, placebo,
     guarded, the second baseline sample, dose wordings if asked);
  2. every two-turn condition, built from that question's `baseline` row.
All two-turn conditions of a question therefore continue the SAME first
answer, which the Claude Code route could not do.

Rows have the same shape as `fsyc.run`, so `fsyc.analyze` reads them as is.
Resumable: finished rows are skipped, and submitted batch ids are kept in
`<out>.batches.json`, so an interrupted run waits for its batch instead of
paying for it twice. Errored or expired requests are not written and are
retried on the next invocation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .backends import AnthropicBackend, parse_probability
from .data import load_jsonl
from .prompts import ALL_CONDITIONS, EXTRA_CONDITIONS, SINGLE_TURN, second_turn
from .run import TWO_TURN, _load_done, placebo_number

# Addendum L, core set. Dose wordings are optional (`--conditions core,dose`).
CORE_SINGLE = ALL_CONDITIONS[:-1] + ["baseline_rep"]       # all but "pushback"
CORE_TWO = list(TWO_TURN)


def _cid(i: int, c: str) -> str:
    return f"q{i}-{c}"          # custom_id: <= 64 chars, [A-Za-z0-9_-]


def plan(questions, conditions, done, backend):
    """Requests still needed for this round: {custom_id: (q, condition, params, extra)}."""
    single = [c for c in conditions if c in SINGLE_TURN]
    two = [c for c in conditions if c in TWO_TURN]
    r1, r2 = {}, {}
    for i, q in enumerate(questions):
        q = dict(q, placebo_number=placebo_number(q["id"]))
        for c in single:
            if (q["id"], c) not in done:
                r1[_cid(i, c)] = (q, c, backend.params(SINGLE_TURN[c](q)), {})
        first = done.get((q["id"], "baseline"))
        if first is None or first["p"] is None:
            continue
        for c in two:
            if (q["id"], c) in done:
                continue
            st = second_turn(c, q, first["raw"], first["p"])
            if st is not None:
                msgs, extra = st
                r2[_cid(i, c)] = (q, c, backend.params(msgs), extra)
    return r1, r2


def _row(q, c, backend, text, meta, extra, batch_id):
    return {"id": q["id"], "condition": c, "model": backend.name, "p": parse_probability(text),
            "outcome": q["outcome"], "market_price": q["market_price"], "source": q["source"],
            "raw": text, **extra, **meta, "effort": backend.effort, "batch_id": batch_id}


def submit_and_collect(client, reqs, backend, out: Path, state_path: Path, log=print, poll=30.0):
    """Submit one batch (or resume a pending one), wait, append the rows."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {"pending": None, "done": []}
    if state["pending"]:
        bid = state["pending"]
        log(f"resuming batch {bid}")
    else:
        b = client.messages.batches.create(
            requests=[{"custom_id": k, "params": p} for k, (_, _, p, _) in reqs.items()])
        bid = b.id
        state["pending"] = bid
        state_path.write_text(json.dumps(state))
        log(f"submitted batch {bid}: {len(reqs)} requests")
    while True:
        b = client.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        log(f"batch {bid}: {b.request_counts}")
        time.sleep(poll)
    n_ok = n_bad = 0
    with out.open("a") as f:
        for r in client.messages.batches.results(bid):
            if r.custom_id not in reqs:
                continue           # submitted by an earlier invocation and already written
            q, c, _, extra = reqs[r.custom_id]
            if r.result.type != "succeeded":
                n_bad += 1
                continue
            text, meta = AnthropicBackend.read(r.result.message)
            f.write(json.dumps(_row(q, c, backend, text, meta, extra, bid)) + "\n")
            n_ok += 1
    state["pending"] = None
    state["done"].append(bid)
    state_path.write_text(json.dumps(state))
    log(f"batch {bid}: {n_ok} rows written, {n_bad} errored/expired (rerun to retry)")
    return n_ok


def run_batches(questions, backend, out: Path, conditions, client, log=print, poll=30.0):
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(".batches.json")
    total = 0
    for rnd in (1, 2):
        done = _load_done(out)
        r1, r2 = plan(questions, conditions, done, backend)
        reqs = r1 if rnd == 1 else r2
        if reqs or (state_path.exists() and json.loads(state_path.read_text())["pending"]):
            log(f"round {rnd}: {len(reqs)} requests")
            total += submit_and_collect(client, reqs, backend, out, state_path, log, poll)
    return total


def expand(conds):
    out = []
    for c in conds:
        if c == "core":
            out += CORE_SINGLE + CORE_TWO
        elif c == "dose":
            out += EXTRA_CONDITIONS
        else:
            out.append(c)
    bad = set(out) - set(SINGLE_TURN) - set(TWO_TURN)
    if bad:
        raise SystemExit(f"unknown conditions {bad}")
    return list(dict.fromkeys(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions_post_2026-06.jsonl")
    ap.add_argument("--model", required=True, help="claude-opus-5-5 | claude-fable-5-2 | ...")
    ap.add_argument("--effort", default=None, help="default: the model's API default, set explicitly")
    ap.add_argument("--conditions", default="core", help="'core', 'dose', or a comma list")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    be = AnthropicBackend(a.model, effort=a.effort, client=client)
    qs = load_jsonl(a.questions)[: a.limit]
    out = Path(a.out or f"results/api__{a.model}.jsonl")
    n = run_batches(qs, be, out, expand(a.conditions.split(",")), client)
    print(f"done: {n} new rows -> {out}")


if __name__ == "__main__":
    main()
