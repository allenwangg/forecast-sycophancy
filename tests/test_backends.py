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


def test_claude_code_backend_commands(monkeypatch, tmp_path):
    import json
    import subprocess

    from fsyc.backends import ClaudeCodeBackend
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        out = {"is_error": False, "result": "x\nProbability: 0.4", "session_id": "S1",
               "usage": {"input_tokens": 5, "output_tokens": 6}, "modelUsage": {"m": {}}, "num_turns": 1}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-passed")
    be = ClaudeCodeBackend("opus", workdir=str(tmp_path))
    text, meta = be.chat_meta([{"role": "system", "content": "GUARD"}, {"role": "user", "content": "Q"}])
    cmd, kw = calls[-1]
    assert cmd[:3] == ["claude", "-p", "Q"] and cmd[cmd.index("--system-prompt") + 1] == "GUARD"
    assert cmd[cmd.index("--tools") + 1] == "" and "--resume" not in cmd
    assert "ANTHROPIC_API_KEY" not in kw["env"]
    assert meta["session_id"] == "S1" and meta["served_models"] == ["m"]
    be.chat_meta([{"role": "user", "content": "Q"}])
    assert calls[-1][0][calls[-1][0].index("--system-prompt") + 1] == be.NEUTRAL_SYSTEM
    be.chat_resume({"session_id": "S1"}, "I disagree")
    cmd = calls[-1][0]
    assert cmd[2] == "I disagree" and cmd[cmd.index("--resume") + 1] == "S1"


def test_claude_code_backend_raises_on_cli_error(monkeypatch, tmp_path):
    import json
    import subprocess

    import pytest

    from fsyc.backends import ClaudeCodeBackend
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, json.dumps({"is_error": True, "result": "Failed to authenticate"}), ""))
    with pytest.raises(RuntimeError):
        ClaudeCodeBackend(workdir=str(tmp_path)).chat([{"role": "user", "content": "Q"}])


def test_clean_env_backend_strips_session_context(monkeypatch, tmp_path):
    from fsyc.backends import ClaudeCodeBackend, make_backend
    b = make_backend("claude-code-clean:opus", workdir=str(tmp_path))
    assert b.name == "claude-code-clean:opus" and b.clean_env
    keep = ClaudeCodeBackend.CLEAN_ENV_KEEP
    assert keep.match("HTTPS_PROXY") and keep.match("PATH")
    for k in ("CLAUDE_MEMORY_STORES", "CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", "CLAUDE_CODE_USER_EMAIL",
              "ANTHROPIC_API_KEY", "CLAUDE_CODE_ARTIFACT_TYPES"):
        assert not keep.match(k)


def test_train_pilot_examples_target_own_answer():
    from fsyc.train_pilot import make_examples, split
    qs = [{"id": f"q{i}", "question": f"Q{i}?", "freeze_date": "2025-05-01", "market_price": 0.3}
          for i in range(20)]
    res = []
    for q in qs:
        res.append({"id": q["id"], "condition": "baseline", "p": 0.4, "raw": "base\nProbability: 0.4"})
        res.append({"id": q["id"], "condition": "market", "p": 0.32, "raw": "mkt\nProbability: 0.32"})
    res[0]["p"] = None  # unparsed baseline -> question dropped
    ex = make_examples(qs, res)
    assert len(ex) == 19 and all(len(v) == 4 for v in ex.values())
    yes, mkt = ex["q1"][1], ex["q1"][3]
    assert "resolves YES" in yes[0]["content"] and yes[-1]["content"] == "base\nProbability: 0.4"
    assert "trading at 30%" in mkt[0]["content"] and mkt[-1]["content"].endswith("0.32")
    tr, va = split(ex)
    tq = {m["messages"][0]["content"].split("Question: ")[1].split("\n")[0] for m in tr}
    vq = {m["messages"][0]["content"].split("Question: ")[1].split("\n")[0] for m in va}
    assert not tq & vq and len(tr) + len(va) == 76


def test_mlx_backend_spec():
    from fsyc.backends import make_backend
    b = make_backend("mlx:gemma3-4b-lora@8081")
    assert b.name == "mlx:gemma3-4b-lora" and b.port == 8081
