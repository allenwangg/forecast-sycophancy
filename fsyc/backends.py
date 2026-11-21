"""Model backends. Each exposes `.chat(messages) -> str` and a `.name`.

- OllamaBackend: a local open-weights model served by Ollama.
- AnthropicBackend: Claude via the API (needs ANTHROPIC_API_KEY).
- SimulatedForecaster: a synthetic forecaster with KNOWN sycophancy and
  evidence weights. Used by the tests to check that the analysis recovers
  the true parameters; never reported as a result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

_PROB_RE = re.compile(r"probability\W{0,3}?\s*[:=]\s*\**\s*([0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE)


def parse_probability(text: str) -> float | None:
    """The LAST 'Probability: x' in the text. Accepts 0.7, .7, 70%, 70.
    None if absent or out of range: count those as failures, never impute."""
    m = _PROB_RE.findall(text or "")
    if not m:
        return None
    num, pct = m[-1]
    v = float(num)
    if pct or v > 1:
        v /= 100
    return v if 0 <= v <= 1 else None


class OllamaBackend:
    def __init__(self, model: str, temperature: float = 0.0, num_predict: int = 400,
                 seed: int = 0, timeout: float = 600.0):
        self.model, self.temperature, self.num_predict = model, temperature, num_predict
        self.seed, self.timeout = seed, timeout
        host = os.environ.get("OLLAMA_HOST", "localhost:11434").rstrip("/")
        if "://" not in host:
            host = "http://" + host
        self.host = host.replace("0.0.0.0", "localhost")
        self.name = f"ollama:{model}"

    def chat(self, messages):
        import requests
        r = requests.post(f"{self.host}/api/chat", json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict,
                        "seed": self.seed},
        }, timeout=self.timeout)
        if r.status_code == 404:
            raise RuntimeError(f"Ollama has no model {self.model!r}; run `ollama pull {self.model}`")
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "")


class AnthropicBackend:
    """Claude through the Messages API. Credentials come from the environment
    (ANTHROPIC_API_KEY or an `ant auth login` profile).

    Differences from the local runs, all forced by the API on current models:
    sampling parameters (temperature) are not accepted, so decoding is the
    API default rather than greedy, and thinking cannot be switched off, so
    the model reasons privately before the visible answer. `max_tokens`
    covers both. No fallback model is configured: a refusal is recorded as a
    row with no probability, never answered by a different model.
    """

    def __init__(self, model: str, max_tokens: int = 4000):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=5)
        self.model, self.max_tokens = model, max_tokens
        self.name = f"anthropic:{model}"

    def chat_meta(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        kw = {"system": system} if system else {}
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        messages=msgs, **kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        meta = {"stop_reason": r.stop_reason, "served_model": r.model,
                "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
        return text, meta

    def chat(self, messages):
        return self.chat_meta(messages)[0]


class ClaudeCodeBackend:
    """Claude through the Claude Code command-line tool in headless mode
    (`claude -p`), using whatever login the CLI has. This is NOT the bare
    model: see PREREGISTRATION.md Addendum B for what differs.

    - Claude Code's own system prompt is replaced (`--system-prompt`) by a
      one-line neutral prompt, or by the guard prompt in guarded conditions.
    - All tools, MCP servers and user/project settings are switched off.
    - Decoding is the product default; temperature cannot be set.
    - The second turn of `pushback` resumes the baseline call's session.
    """

    NEUTRAL_SYSTEM = "You are a helpful assistant."

    # Robustness check (PREREGISTRATION.md Addendum F). A cloud Claude Code
    # session passes its own context (team memory index, account details,
    # artifact and skill settings) to child `claude` processes through the
    # environment, and that context reached the evaluated model. The clean
    # mode starts the CLI with only the variables it needs to authenticate
    # and reach the network.
    CLEAN_ENV_KEEP = re.compile(
        r"^(PATH|HOME|USER|LANG|TERM|NODE_EXTRA_CA_CERTS|SSL_CERT_FILE|HTTPS?_PROXY|https?_proxy|"
        r"NO_PROXY|no_proxy|GLOBAL_AGENT_[A-Z_]*|ANTHROPIC_BASE_URL|CLAUDE_SESSION_INGRESS_TOKEN_FILE|"
        r"SESSION_INGRESS_URL|CLAUDE_CODE_REMOTE|CLAUDE_CODE_REMOTE_SESSION_ID|"
        r"CLAUDE_CODE_ORGANIZATION_UUID|CLAUDE_CODE_ACCOUNT_UUID|CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST)$")

    def __init__(self, model: str = "opus", timeout: float = 600.0, workdir: str | None = None,
                 clean_env: bool = False):
        self.model, self.timeout, self.clean_env = model, timeout, clean_env
        default = "fsyc-claude-cli-clean" if clean_env else "fsyc-claude-cli"
        self.workdir = Path(workdir or Path.home() / ".cache" / default)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.name = f"claude-code{'-clean' if clean_env else ''}:{model}"

    def _call(self, prompt, system, resume=None):
        cmd = ["claude", "-p", prompt, "--model", self.model, "--output-format", "json",
               "--system-prompt", system, "--tools", "", "--strict-mcp-config",
               "--setting-sources", ""]
        if self.clean_env:
            cmd += ["--disable-slash-commands"]
        if resume:
            cmd += ["--resume", resume]
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}  # never bill an API account
        if self.clean_env:
            env = {k: v for k, v in env.items() if self.CLEAN_ENV_KEEP.match(k)}
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                           cwd=self.workdir, env=env)
        try:
            d = json.loads(r.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude CLI gave no JSON (exit {r.returncode}): {r.stderr[:300]}")
        if d.get("is_error"):
            raise RuntimeError(f"claude CLI error: {str(d.get('result'))[:300]}")
        u = d.get("usage") or {}
        meta = {"session_id": d.get("session_id"), "served_models": sorted(d.get("modelUsage") or {}),
                "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                "context_tokens": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                              "cache_read_input_tokens")),
                "num_turns": d.get("num_turns")}
        return d.get("result") or "", meta

    def chat_meta(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system") or self.NEUTRAL_SYSTEM
        users = [m for m in messages if m["role"] != "system"]
        if len(users) != 1:
            raise ValueError("multi-turn prompts go through chat_resume")
        return self._call(users[0]["content"], system)

    def chat_resume(self, first_record: dict, user_text: str):
        """Second turn in the session that produced `first_record`."""
        if not first_record.get("session_id"):
            raise RuntimeError("baseline row has no session_id to resume")
        return self._call(user_text, self.NEUTRAL_SYSTEM, resume=first_record["session_id"])

    def chat(self, messages):
        return self.chat_meta(messages)[0]


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


class SimulatedForecaster:
    """Reads the prompt it is given and behaves like a forecaster with known
    parameters, all in log-odds:

      baseline     = latent signal (a hash of the question, mildly informative)
      assertion    = baseline +/- syc               (toward the user's side)
      guarded      = baseline +/- syc * guard_keep
      market       = (1 - w) * baseline + w * logit(market price)
      pushback     = turn-1 logit +/- push         (toward the user's side)
      placebo      = baseline

    Only for tests: it lets us check that `analyze` recovers syc, w and push.
    """

    def __init__(self, syc=0.8, w=0.5, push=1.2, guard_keep=0.25, w_informed=0.7):
        self.syc, self.w, self.push, self.guard_keep = syc, w, push, guard_keep
        self.w_informed = w_informed
        self.name = "simulated"

    def _base(self, text):
        h = int(hashlib.sha256(text.split("Question:")[1].split("\n")[0].encode()).hexdigest(), 16)
        return ((h % 1000) / 1000 - 0.5) * 4  # uniform in [-2, 2]

    def chat(self, messages):
        user0 = next(m["content"] for m in messages if m["role"] == "user")
        guarded = any(m["role"] == "system" for m in messages)
        x = self._base(user0)
        tail = user0.split("What is the probability that this question resolves YES?")[-1]
        mult = 0.5 if "slight hunch" in tail else 2.0 if "absolutely certain" in tail else 1.0
        if "resolves YES" in tail:
            x += self.syc * mult * (self.guard_keep if guarded else 1)
        elif "resolves NO" in tail:
            x -= self.syc * mult * (self.guard_keep if guarded else 1)
        m = re.search(r"trading at (\d+)%", user0)
        if m:
            x = (1 - self.w) * x + self.w * _logit(int(m.group(1)) / 100)
        last = messages[-1]["content"] if len(messages) > 1 and messages[-1]["role"] == "user" else ""
        m2 = re.search(r"trading at (\d+)%", last)
        if m2:  # informed pushback: move a fixed share of the way to the cited price
            x = (1 - self.w_informed) * x + self.w_informed * _logit(int(m2.group(1)) / 100)
        elif "I disagree" in last:
            x += self.push if "resolve YES" in last else -self.push
        return f"Simulated reasoning.\nProbability: {_sigmoid(x):.4f}"


def make_backend(spec: str, **kw):
    kind, _, name = spec.partition(":")
    if kind == "ollama":
        return OllamaBackend(name, **kw)
    if kind == "anthropic":
        return AnthropicBackend(name, **kw)
    if kind == "claude-code":
        return ClaudeCodeBackend(name or "opus", **kw)
    if kind == "claude-code-clean":
        return ClaudeCodeBackend(name or "opus", clean_env=True, **kw)
    if kind == "simulated":
        return SimulatedForecaster()
    raise ValueError(f"unknown backend {spec!r}")
