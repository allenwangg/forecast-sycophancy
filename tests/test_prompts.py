from fsyc import prompts as P

Q = {"id": "x", "question": "Will X happen?", "background": "bg", "resolution_criteria": "rc",
     "freeze_date": "2026-01-01", "close_date": "2026-06-01", "market_price": 0.23, "placebo_number": 41}


def _text(msgs):
    return "\n".join(m["content"] for m in msgs)


def test_each_condition_adds_one_thing_to_baseline():
    base = P.baseline(Q)[0]["content"]
    core = P.core(Q)
    for name, fn in P.SINGLE_TURN.items():
        msgs = fn(Q)
        user = msgs[-1]["content"]
        assert user.startswith(core) and user.endswith(P.INSTRUCTIONS), name
        if name != "baseline":
            assert _text(msgs) != base, name


def test_guard_is_only_difference():
    assert P.guarded_assert_yes(Q)[1:] == P.assert_yes(Q)
    assert P.guarded_market(Q)[0]["role"] == "system"


def test_market_shows_real_price():
    assert "23%" in P.market(Q)[0]["content"]


def test_pushback_goes_against_model_lean():
    assert P.pushback_side(0.8) == "NO"
    assert P.pushback_side(0.2) == "YES"
    msgs = P.pushback(Q, "reasoning\nProbability: 0.8", 0.8)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "resolve NO" in msgs[-1]["content"]
