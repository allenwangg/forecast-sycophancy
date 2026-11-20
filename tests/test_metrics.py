import numpy as np
import pytest

from fsyc import metrics as M
from fsyc.backends import parse_probability


def test_brier_known_values():
    assert M.brier([1, 0], [1, 0]) == 0
    assert M.brier([0.5, 0.5], [1, 0]) == 0.25
    assert M.brier([0, 1], [1, 0]) == 1


def test_log_loss_half():
    assert M.log_loss([0.5], [1]) == pytest.approx(np.log(2))


def test_ece_perfectly_calibrated():
    # forecasts of 0.2 where 20% resolve YES
    p = [0.2] * 10
    y = [1, 1] + [0] * 8
    assert M.ece(p, y) == pytest.approx(0)


def test_murphy_exact_when_constant_within_bins():
    p = [0.2] * 10 + [0.8] * 10
    y = [1, 1] + [0] * 8 + [1] * 7 + [0] * 3
    d = M.murphy_decomposition(p, y)
    assert d["residual"] == pytest.approx(0, abs=1e-12)


def test_auroc():
    assert M.auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1
    assert M.auroc([0.5, 0.5], [1, 0]) == 0.5


def test_paired_diff_zero_for_identical():
    rng = np.random.default_rng(1)
    p = rng.random(50)
    y = (rng.random(50) < p).astype(int)
    d, lo, hi = M.paired_bootstrap_diff(M.brier, p, p, y, n_boot=200)
    assert d == lo == hi == 0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        M.brier([1.2], [1])
    with pytest.raises(ValueError):
        M.brier([0.5], [2])


@pytest.mark.parametrize("text,expected", [
    ("blah\nProbability: 0.73", 0.73),
    ("Probability: .4", 0.4),
    ("probability = 65%", 0.65),
    ("Probability: 30", 0.30),
    ("Probability: 0.2 ... on reflection\nProbability: 0.35", 0.35),
    ("no number here", None),
    ("Probability: 250", None),
])
def test_parse_probability(text, expected):
    got = parse_probability(text)
    assert got == (pytest.approx(expected) if expected is not None else None)
