"""Compare models on the same questions: a table of the headline statistics
and paired bootstrap CIs for the between-model difference in assertion gap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analyze as A


def gap_difference(w_a, w_b):
    """gap(b) - gap(a) on the questions both models answered, paired bootstrap."""
    ga = (w_a["assert_yes"] - w_a["assert_no"]).dropna()
    gb = (w_b["assert_yes"] - w_b["assert_no"]).dropna()
    ids = ga.index.intersection(gb.index)
    x = (gb.loc[ids] - ga.loc[ids]).to_numpy()
    return A._boot(lambda i: x[i].mean(), len(x))


def table(stats_list):
    def f(v, pct=False):
        m = 100 if pct else 1
        d = 0 if pct else 2
        return f"{v['est'] * m:.{d}f} [{v['lo'] * m:.{d}f}, {v['hi'] * m:.{d}f}]"
    rows = [
        ("Opinion gap, p(says YES) − p(says NO)", lambda s: f(s["H1_assertion_gap"])),
        ("Implied likelihood ratio of one opinion", lambda s: f(s["H1_implied_likelihood_ratio"])),
        ("'I disagree': mean move toward user", lambda s: f(s["H3_pushback"]["shift_toward_user_p"])),
        ("'I disagree': answers crossing 50% (%)", lambda s: f(s["H3_pushback"]["flip_rate"], True)),
        ("Weight on real market price (0 to 1)", lambda s: f(s["H2_market_weight"])),
        ("Brier change when user is wrong", lambda s: f(s["H4_brier_user_wrong_minus_baseline"])),
        ("Opinion gap with guard prompt", lambda s: f(s["H5_guarded_assertion_gap"])),
        ("Market weight with guard prompt", lambda s: f(s["H5_guarded_market_weight"])),
        ("Placebo: mean absolute change", lambda s: f(s["placebo_mean_abs_shift"])),
    ]
    names = [s["model"].replace("ollama:", "") for s in stats_list]
    out = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(fn(s) for s in stats_list) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="results/comparison.md")
    a = ap.parse_args(argv)
    res = [A.analyze(A.load(p)) for p in a.results]
    txt = [table([r["stats"] for r in res]), ""]
    cal = ["| Model | Baseline Brier | Baseline AUROC | Brier with market price |", "|---|---|---|---|"]
    for r in res:
        c = r["calibration"]
        cal.append(f"| {r['stats']['model'].replace('ollama:', '')} | {c.loc['baseline', 'brier']:.3f} | "
                   f"{c.loc['baseline', 'auroc']:.3f} | {c.loc['market', 'brier']:.3f} |")
    txt += cal + [""]
    diffs = {}
    for i in range(len(res) - 1):
        d = gap_difference(res[i]["wide"], res[i + 1]["wide"])
        k = f"{res[i + 1]['stats']['model']} minus {res[i]['stats']['model']}"
        diffs[k] = d
        txt.append(f"Opinion gap, {k}: {d['est']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] (n={d['n']}, paired)")
    Path(a.out).write_text("\n".join(txt) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(diffs, indent=2))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
