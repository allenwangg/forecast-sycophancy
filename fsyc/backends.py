"""Model backends. Each exposes `.chat(messages) -> str` and a `.name`.

- OllamaBackend: a local open-weights model served by Ollama.
- AnthropicBackend: Claude via the API (needs ANTHROPIC_API_KEY).
- SimulatedForecaster: a synthetic forecaster with KNOWN sycophancy and
  evidence weights. Used by the tests to check that the analysis recovers
  the true parameters; never reported as a result.
"""
from __future__ import annotations

import hashlib
import math
import os
import re

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
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 600):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"anthropic:{model}"

    def chat(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        kw = {"system": system} if system else {}
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        temperature=self.temperature, messages=msgs, **kw)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


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

    def __init__(self, syc=0.8, w=0.5, push=1.2, guard_keep=0.25):
        self.syc, self.w, self.push, self.guard_keep = syc, w, push, guard_keep
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
        if len(messages) > 1 and messages[-1]["role"] == "user" and "I disagree" in messages[-1]["content"]:
            x += self.push if "resolve YES" in messages[-1]["content"] else -self.push
        return f"Simulated reasoning.\nProbability: {_sigmoid(x):.4f}"


def make_backend(spec: str, **kw):
    kind, _, name = spec.partition(":")
    if kind == "ollama":
        return OllamaBackend(name, **kw)
    if kind == "anthropic":
        return AnthropicBackend(name, **kw)
    if kind == "simulated":
        return SimulatedForecaster()
    raise ValueError(f"unknown backend {spec!r}")
