"""The Claude backend is tested against a fake SDK: no network, no key."""
import sys
import types

from fsyc.backends import make_backend


class _Block:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


class _Resp:
    stop_reason = "end_turn"
    model = "fake-model"
    usage = types.SimpleNamespace(input_tokens=11, output_tokens=7)
    content = [_Block("thinking"), _Block("text", "Because.\nProbability: 0.3")]


def test_anthropic_backend_request_shape(monkeypatch):
    seen = {}

    class _Client:
        def __init__(self, **kw):
            self.messages = types.SimpleNamespace(create=lambda **k: seen.update(k) or _Resp())

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    be = make_backend("anthropic:some-model")
    text, meta = be.chat_meta([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert text.endswith("Probability: 0.3")           # thinking blocks are not part of the answer
    assert seen["system"] == "S" and seen["messages"] == [{"role": "user", "content": "U"}]
    assert "temperature" not in seen and "fallbacks" not in seen
    assert meta == {"stop_reason": "end_turn", "served_model": "fake-model",
                    "input_tokens": 11, "output_tokens": 7}
