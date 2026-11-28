"""The batch runner and Experiment 5, against a fake Batches API that answers
with the simulated forecaster. No network, no key."""
import json
import re
import types

import pytest

from fsyc import analyze as A
from fsyc.api_batch import CORE_SINGLE, CORE_TWO, expand, run_batches
from fsyc.backends import AnthropicBackend, SimulatedForecaster, _logit, _sigmoid
from fsyc.prompts import false_same_price


def _questions(n=150):
    return [{"id": f"q{i}", "source": "manifold", "question": f"Will thing {i} happen?",
             "background": "", "resolution_criteria": "", "freeze_date": "2026-07-01",
             "close_date": "2026-12-01", "market_price": 0.1 + 0.8 * ((i * 37) % 100) / 100,
             "outcome": i % 2} for i in range(n)]


class Discerning(SimulatedForecaster):
    """Moves w_true of the way to a quote that equals the real market price
    and w_false of the way to any other quote."""

    def __init__(self, markets, w_true=0.7, w_false=0.3):
        super().__init__()
        self.markets, self.w_true, self.w_false = markets, w_true, w_false

    def chat(self, messages):
        last = messages[-1]["content"]
        two_turn = [m["role"] for m in messages] == ["user", "assistant", "user"]
        m = re.search(r"trading at (\d+)%", last) if two_turn else None
        if not m:
            return super().chat(messages)
        q = messages[0]["content"].split("Question: ")[1].split("\n")[0]
        p1 = float(messages[1]["content"].rsplit("Probability: ", 1)[1])
        cited = int(m.group(1)) / 100
        true = abs(cited - round(self.markets[q], 2)) < 0.006
        w = self.w_true if true else self.w_false
        x = (1 - w) * _logit(p1) + w * _logit(cited)
        return f"Simulated.\nProbability: {_sigmoid(x):.4f}"


class FakeBatches:
    def __init__(self, sim):
        self.sim, self.store, self.created = sim, {}, 0

    def create(self, requests):
        self.created += 1
        bid = f"b{self.created}"
        self.store[bid] = requests
        return types.SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return types.SimpleNamespace(processing_status="ended", request_counts={})

    def results(self, bid):
        for r in self.store[bid]:
            p = r["params"]
            msgs = ([{"role": "system", "content": p["system"]}] if "system" in p else []) + p["messages"]
            text = self.sim.chat(msgs)
            msg = types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                model="fake", stop_details=None, usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))
            yield types.SimpleNamespace(custom_id=r["custom_id"],
                                        result=types.SimpleNamespace(type="succeeded", message=msg))


def _client(sim):
    return types.SimpleNamespace(messages=types.SimpleNamespace(batches=FakeBatches(sim)))


def _run(tmp_path, sim, qs):
    client = _client(sim)
    be = AnthropicBackend("claude-opus-5-5", client=client)
    out = tmp_path / "api.jsonl"
    n = run_batches(qs, be, out, expand(["core"]), client, log=lambda *a: None, poll=0)
    return client, be, out, n


def test_two_rounds_one_first_turn_and_resume(tmp_path):
    qs = _questions(40)
    client, be, out, n = _run(tmp_path, SimulatedForecaster(), qs)
    assert client.messages.batches.created == 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["condition"] for r in rows} >= set(CORE_SINGLE) | {"pushback", "pushback_informed"}
    base = {r["id"]: r["p"] for r in rows if r["condition"] == "baseline"}
    for r in rows:
        if r["condition"] in CORE_TWO:
            assert r["p_first"] == base[r["id"]]        # every second turn continues the same first answer
    assert all(r["effort"] == "medium" and r["batch_id"] for r in rows)
    # resume: nothing left, no new batch
    assert run_batches(qs, be, out, expand(["core"]), client, log=lambda *a: None, poll=0) == 0
    assert client.messages.batches.created == 2


def test_request_has_no_sampling_params(tmp_path):
    client, *_ = _run(tmp_path, SimulatedForecaster(), _questions(5))
    for reqs in client.messages.batches.store.values():
        for r in reqs:
            assert len(r["custom_id"]) <= 64 and re.fullmatch(r"[A-Za-z0-9_-]+", r["custom_id"])
            p = r["params"]
            assert "temperature" not in p and "fallbacks" not in p and p["output_config"] == {"effort": "medium"}


def test_false_same_price_points_same_way_and_is_false():
    for i in range(200):
        p1, m = (i % 19 + 1) / 20, ((i * 7) % 19 + 1) / 20
        c = false_same_price(f"q{i}", p1, m)
        if c is not None:
            assert (c - p1) * (m - p1) > 0 and abs(c - m) >= 0.05


def test_e5_no_discrimination_gives_zero(tmp_path):
    _, _, out, _ = _run(tmp_path, SimulatedForecaster(w_informed=0.6), _questions())
    s = A.analyze(A.load(out))["stats"]
    e5 = s["E5_decoupled_quote"]
    assert e5["weight_true_minus_false_same"]["est"] == pytest.approx(0, abs=0.03)
    assert s["baseline_test_retest_mean_abs_diff"]["est"] == 0       # simulated forecaster is deterministic


def test_e5_detects_discrimination(tmp_path):
    qs = _questions()
    sim = Discerning({q["question"]: q["market_price"] for q in qs}, w_true=0.7, w_false=0.3)
    _, _, out, _ = _run(tmp_path, sim, qs)
    e5 = A.analyze(A.load(out))["stats"]["E5_decoupled_quote"]
    assert e5["weight_true_minus_false_same"]["est"] == pytest.approx(0.4, abs=0.05)
    assert e5["weight_true_minus_false_same"]["lo"] > 0.2
    assert e5["fraction_moved_regression"]["false"]["hi"] < 0
