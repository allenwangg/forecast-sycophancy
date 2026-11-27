"""Experiment 5 on the subscription route (PREREGISTRATION.md Addendum M):
combine the new same-direction false quotes with the existing clean true
and mirrored-false quotes on the same questions, then report."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import analyze as A


def report(model: str) -> dict:
    main = A.load(f"results/claude-code-clean__{model}.jsonl")
    new = A.load(f"results/claude-code-clean-e5__{model}.jsonl")
    df = pd.concat([main[main["condition"].isin(["pushback_informed", "pushback_misinformed"])], new],
                   ignore_index=True)
    out = A.decoupled_quote_stats(df) or {}
    d = new[(new["condition"] == "pushback_false_same") & new["p"].notna()]
    move = (d["p"] - d["p_first"]) * np.sign(d["cited_price"] - d["p_first"])
    b1 = (d["p_first"] - d["outcome"]) ** 2
    b2 = (d["p"] - d["outcome"]) ** 2
    out["false_same_rows"] = int(len(d))
    out["false_same_parse_failures"] = int(new["p"].isna().sum())
    out["false_same_mean_move"] = float(move.mean())
    out["false_same_brier_change"] = float((b2 - b1).mean())
    out["false_same_unchanged_share"] = float((move == 0).mean())
    return out


if __name__ == "__main__":
    res = {m: report(m) for m in sys.argv[1:] or ["opus", "fable"]}
    print(json.dumps(res, indent=1, default=float))
