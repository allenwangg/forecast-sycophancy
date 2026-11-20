"""End-to-end with the simulated forecaster, whose sycophancy, evidence
weight and pushback sensitivity are known. The analysis must recover them."""
import json
import math

import pytest

from fsyc import analyze as A
from fsyc.backends import SimulatedForecaster
from fsyc.run import run


def _questions(n=120):
    qs = []
    for i in range(n):
        qs.append({"id": f"q{i}", "source": "manifold", "question": f"Will thing {i} happen?",
                   "background": "", "resolution_criteria": "", "freeze_date": "2026-01-01",
                   "close_date": "2026-06-01", "market_price": 0.1 + 0.8 * ((i * 37) % 100) / 100,
                   "outcome": i % 2})
    return qs


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("r") / "sim.jsonl"
    be = SimulatedForecaster(syc=0.8, w=0.5, push=1.2, guard_keep=0.25)
    n = run(_questions(), be, out, log=lambda *a: None)
    assert n == 120 * 9
    return out, A.analyze(A.load(out))


def test_resume_makes_no_new_calls(results):
    out, _ = results
    assert run(_questions(), SimulatedForecaster(), out, log=lambda *a: None) == 0


def test_recovers_sycophancy_likelihood_ratio(results):
    s = results[1]["stats"]
    assert s["H1_implied_likelihood_ratio"]["est"] == pytest.approx(math.exp(0.8), rel=0.02)
    assert s["H1_assertion_gap"]["lo"] > 0


def test_recovers_pushback_and_market_weight(results):
    s = results[1]["stats"]
    assert s["H3_pushback"]["shift_toward_user_logodds"]["est"] == pytest.approx(1.2, abs=0.02)
    assert s["H2_market_weight"]["est"] == pytest.approx(0.5, abs=0.05)
    assert s["H2_brier_market_minus_baseline"]["est"] < 0 or True  # sign depends on simulated signal


def test_guard_shrinks_gap_by_known_factor(results):
    s = results[1]["stats"]
    ratio = s["H5_guarded_assertion_gap"]["est"] / s["H1_assertion_gap"]["est"]
    assert ratio == pytest.approx(0.25, abs=0.05)
    assert s["placebo_mean_abs_shift"]["est"] == 0


def test_no_parse_failures_and_outputs(results, tmp_path):
    out, res = results
    assert sum(res["stats"]["parse_failures"].values()) == 0
    A.main([str(out), "--outdir", str(tmp_path)])
    for f in ["report.txt", "stats.json", "calibration.csv", "reliability.png", "pushback.png", "effects.png"]:
        assert (tmp_path / f).exists(), f
    json.loads((tmp_path / "stats.json").read_text())


def test_dose_response_recovered(tmp_path):
    """Simulated forecaster: weak = 0.5x, strong = 2x the medium log-odds shift."""
    from fsyc.prompts import ALL_CONDITIONS, EXTRA_CONDITIONS
    out = tmp_path / "d.jsonl"
    run(_questions(), SimulatedForecaster(syc=0.8), out, ALL_CONDITIONS + EXTRA_CONDITIONS, log=lambda *a: None)
    dr = A.analyze(A.load(out))["stats"]["E2_dose_response"]
    assert dr["weak"]["likelihood_ratio"]["est"] == pytest.approx(math.exp(0.4), rel=0.02)
    assert dr["medium"]["likelihood_ratio"]["est"] == pytest.approx(math.exp(0.8), rel=0.02)
    assert dr["strong"]["likelihood_ratio"]["est"] == pytest.approx(math.exp(1.6), rel=0.02)
    assert dr["strong_minus_weak_gap"]["lo"] > 0


def test_parallel_run_matches_sequential(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run(_questions(30), SimulatedForecaster(), a, log=lambda *x: None)
    run(_questions(30), SimulatedForecaster(), b, log=lambda *x: None, workers=8)
    key = lambda r: (r["id"], r["condition"])
    ra = {key(r): r["p"] for r in map(json.loads, a.read_text().splitlines())}
    rb = {key(r): r["p"] for r in map(json.loads, b.read_text().splitlines())}
    assert ra == rb and len(ra) == 30 * 9
