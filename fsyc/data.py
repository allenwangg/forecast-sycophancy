"""Build the question set from ForecastBench (Karger et al., ICLR 2025).

ForecastBench publishes, every two weeks, a set of forecasting questions
(`datasets/question_sets/<date>-llm.json`) and, later, resolution files
(`datasets/resolution_sets/<date>_resolution_set.json`). We use only the
*market* questions (Manifold, Metaculus, Polymarket, Kalshi, INFER) because
they are single yes/no events with a clean 0/1 resolution, and each one
carries the market price at the time the question set was frozen. That price
is real, pre-resolution evidence; we use it as the positive control.

Data: https://github.com/forecastingresearch/forecastbench-datasets (CC BY-SA 4.0)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter
from pathlib import Path

MARKET_SOURCES = {"manifold", "metaculus", "polymarket", "kalshi", "infer"}
RAW_BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _final_resolutions(resolution_files):
    """id -> (first resolution_date at which the question was resolved to 0/1, outcome).

    Unresolved market questions are scored by ForecastBench against the
    current market price (a number in (0,1)); we drop those and keep only
    questions that actually resolved YES or NO."""
    seen = {}
    for f in sorted(resolution_files):
        for r in json.load(open(f))["resolutions"]:
            if r["source"] not in MARKET_SOURCES or not isinstance(r["id"], str):
                continue
            if r.get("resolved") and r.get("resolved_to") in (0.0, 1.0):
                prev = seen.get(r["id"])
                if prev is None or r["resolution_date"] < prev[0]:
                    seen[r["id"]] = (r["resolution_date"], int(r["resolved_to"]))
    return seen


def _questions(question_files):
    """id -> question record, from the EARLIEST question set it appears in
    (so the market price and 'today' date are as early as possible)."""
    qs = {}
    for f in sorted(question_files):
        d = json.load(open(f))
        for q in d["questions"]:
            if q["source"] in MARKET_SOURCES and isinstance(q["id"], str) and q["id"] not in qs:
                qs[q["id"]] = dict(q, forecast_due_date=d["forecast_due_date"])
    return qs


def _to_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 else None


def _take(pool, outcome, k, max_share):
    """First k rows with this outcome, no source above max_share of k."""
    cap = max(1, int(round(k * max_share)))
    per, out = Counter(), []
    for r in pool:
        if r["outcome"] == outcome and per[r["source"]] < cap:
            out.append(r)
            per[r["source"]] += 1
            if len(out) == k:
                break
    return out


def build(raw_dir, cutoff: str, n: int, seed: int = 0, balance: bool = True,
          max_source_share: float = 0.4, max_file_date: str = "9999-12-31"):
    """Return (questions, stage_counts).

    Filters, in order:
      1. market source, resolved to exactly 0 or 1;
      2. frozen (question-set date) strictly after the model's training cutoff,
         so neither the question's framing date nor its outcome is in training data;
      3. has a usable market price at freeze time;
      4. de-duplicated on question text;
      5. stratified sample: equal YES and NO (if balance), and no single
         source above `max_source_share` of each half (Polymarket otherwise
         dominates the pool).
    """
    # max_file_date pins the snapshot: ForecastBench keeps adding files, and a
    # later resolution file could change which questions count as resolved.
    def pick(pattern):
        return [f for f in glob.glob(os.path.join(raw_dir, pattern))
                if os.path.basename(f)[:10] <= max_file_date]
    qfiles, rfiles = pick("*-llm.json"), pick("*_resolution_set.json")
    qs, res = _questions(qfiles), _final_resolutions(rfiles)
    stages = Counter()
    stages["market_questions"] = len(qs)
    pool = []
    seen_text = set()
    for i, q in qs.items():
        if i not in res:
            continue
        stages["resolved_0_or_1"] += 1
        if q["freeze_datetime"][:10] <= cutoff:
            continue
        stages["frozen_after_cutoff"] += 1
        mp = _to_float(q.get("freeze_datetime_value"))
        if mp is None:
            continue
        stages["has_market_price"] += 1
        key = q["question"].strip().lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        stages["deduplicated"] += 1
        rdate, y = res[i]
        pool.append({
            "id": i,
            "source": q["source"],
            "question": q["question"],
            "background": (q.get("background") or "")[:1500],
            "resolution_criteria": (q.get("resolution_criteria") or "")[:800],
            "freeze_date": q["freeze_datetime"][:10],
            "close_date": (q.get("market_info_close_datetime") or "")[:10],
            "market_price": mp,
            "outcome": y,
            "resolution_date": rdate,
            "url": q.get("url", ""),
        })
    rng = random.Random(seed)
    pool.sort(key=lambda r: r["id"])
    rng.shuffle(pool)
    if balance:
        out = _take(pool, 1, n // 2, max_source_share) + _take(pool, 0, n - n // 2, max_source_share)
    else:
        out = pool[:n]
    rng.shuffle(out)
    stages["sampled"] = len(out)
    return out, dict(stages)


def download(raw_dir, since: str):
    """Fetch question/resolution files dated >= `since` (YYYY-MM-DD) with the
    GitHub contents API. Needs network; tests never call this."""
    import requests
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    for sub in ("question_sets", "resolution_sets"):
        listing = requests.get(
            f"https://api.github.com/repos/forecastingresearch/forecastbench-datasets/contents/datasets/{sub}",
            timeout=60).json()
        for item in listing:
            name = item["name"]
            if name[:10] < since or not (name.endswith("-llm.json") or name.endswith("_resolution_set.json")):
                continue
            dest = Path(raw_dir) / name
            if dest.exists():
                continue
            r = requests.get(f"{RAW_BASE}/{sub}/{name}", timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print("downloaded", name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw", help="folder with ForecastBench JSON files")
    ap.add_argument("--download-since", default=None, help="download files dated >= this first")
    ap.add_argument("--cutoff", required=True, help="model training cutoff, YYYY-MM-DD")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--max-file-date", default="9999-12-31",
                    help="ignore ForecastBench files dated after this (pins the snapshot)")
    ap.add_argument("--out", default="data/questions.jsonl")
    a = ap.parse_args(argv)
    if a.download_since:
        download(a.raw, a.download_since)
    rows, stages = build(a.raw, a.cutoff, a.n, a.seed, balance=not a.no_balance,
                        max_file_date=a.max_file_date)
    for k, v in stages.items():
        print(f"{k:>22}: {v}")
    print("sources:", dict(Counter(r["source"] for r in rows)))
    write_jsonl(rows, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
