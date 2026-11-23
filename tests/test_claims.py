"""The numbers the README and REPORT state are checked against the committed
results, so a number cannot drift away from the data that produced it.

One did before this test existed: the summary table showed Gemma 3 4B's
weight on a real price as 0.67 when the data say 0.665.
"""
import json
import math
import re
from pathlib import Path

import pytest

from fsyc import analyze as A
from fsyc import summary as S

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = (ROOT / "README.md").read_text()
REPORT = (ROOT / "REPORT.md").read_text()

RUNS = sorted(p.parent.name for p in RESULTS.glob("*/stats.json") if (RESULTS / f"{p.parent.name}.jsonl").exists())


def _points(d, path=()):
    """Every point estimate, count and share in a stats dict. Interval ends
    are left out: they depend on the number of bootstrap draws."""
    out = {}
    for k, v in d.items():
        if k in ("lo", "hi"):
            continue
        p = path + (k,)
        if isinstance(v, dict):
            out.update(_points(v, p))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p] = float(v)
    return out


def _get(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


@pytest.mark.parametrize("run", RUNS)
def test_committed_stats_reproduce_from_the_raw_rows(run, monkeypatch):
    monkeypatch.setattr(A, "N_BOOT", 20)  # point estimates do not depend on it
    fresh = _points(A.analyze(A.load(RESULTS / f"{run}.jsonl"))["stats"])
    committed = _points(json.loads((RESULTS / run / "stats.json").read_text()))
    assert set(committed) <= set(fresh), sorted(set(committed) - set(fresh))
    for p, v in committed.items():
        if math.isnan(v):
            assert math.isnan(fresh[p]), p
        else:
            assert fresh[p] == pytest.approx(v, abs=1e-9), p


FALSE_PRICE = {"claude-code-clean__opus": (183, 185), "claude-code-clean__fable": (183, 184),
               "claude-code-clean-ext__opus": (126, 126), "claude-code-clean-ext__fable": (81, 81)}


def test_false_price_counts_match_the_text():
    assert {run for _, _, run in S.FALSE_PRICE_RUNS} == set(FALSE_PRICE)
    k_all = n_all = 0
    for run, (k, n) in FALSE_PRICE.items():
        c = S.false_price_moves(S.rows(run))
        assert (c["toward"], c["n"], c["away"]) == (k, n, 0), run
        k_all, n_all = k_all + k, n_all + n
    assert (k_all, n_all) == (573, 576)
    for doc in (README, REPORT):
        assert "573 of 576" in doc
        assert "183 of 185" in doc and "183 of 184" in doc


# The README's summary table ("What the experiments add up to"): row label ->
# where the number lives in stats.json. Cells marked † come from Addendum J.
TABLE_ROWS = {
    "A bare opinion": "H1_assertion_gap",
    "More certainty": "E2_dose_response/strong/gap",
    "A claimed track record": "E2_dose_response/expert/gap",
    '"I disagree" in turn 2': "H3_pushback/shift_toward_user_p",
    "A reason with no number": "E2_dose_response/reason/gap",
    "The real market price, in the prompt": "H2_market_weight",
    "The real market price, from the user in turn 2": "E3_informed_user/weight_on_cited_price",
    "A false market price, from the user in turn 2": "E4_misinformed_user/weight_on_false_price",
}
COLUMNS = ["ollama__gemma3__4b", "ollama__gemma3__12b", "ollama__gemma3__27b",
           "claude-code-clean__opus", "claude-code-clean__fable"]
DAGGER = {"claude-code-clean__opus": "claude-code-clean-ext__opus",
          "claude-code-clean__fable": "claude-code-clean-ext__fable"}


def test_readme_summary_table_matches_the_stats():
    section = README.split("## What the experiments add up to", 1)[1].split("What this supports", 1)[0]
    checked = 0
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| What the user") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        label = next(k for k in TABLE_ROWS if cells[0].startswith(k))
        for run, cell in zip(COLUMNS, cells[1:]):
            if cell == "not run":
                continue
            src = DAGGER[run] if "†" in cell else run
            est = _get(S.stats(src), TABLE_ROWS[label])["est"]
            shown = float(re.match(r"[0-9.]+", cell).group())
            assert abs(shown - est) <= 0.005 + 1e-9, (label, run, shown, round(est, 4))
            checked += 1
    assert checked == 35


@pytest.fixture(scope="module")
def summary():
    return S.collect()


def test_summary_file_is_up_to_date(summary):
    assert (RESULTS / "summary.md").read_text() == S.markdown(summary)


def test_report_quotes_the_summary(summary):
    pooled = summary["false_price_pooled"]
    assert f"{pooled['lo']:.1%} to {pooled['hi']:.1%}" in REPORT
    for m in ("Claude Opus 5.5", "Claude Fable 5.2"):
        v = summary["models"][m]
        for key in ("real_weight", "false_weight"):
            assert S.ci(v[key]) in REPORT, (m, key)
        for key in ("real_brier", "false_brier"):
            assert S.ci(v[key], 3, True).replace("-", "−") in REPORT, (m, key)
        assert S.ci(summary["paired"][m], signed=True) in REPORT, m
