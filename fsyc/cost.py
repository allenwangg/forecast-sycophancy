"""Cost estimate for the bare-API study from pilot rows (Addendum L).

Prices are USD per million tokens, first-party API, from the Anthropic
model table as of 2026-11; batch requests are billed at half. Check the
current price page before relying on this."""
from __future__ import annotations

import argparse
import json

PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-fable-5-2": (8.0, 40.0),
          "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def estimate(rows, model, n_calls, batch=True):
    inp = sum(r.get("input_tokens") or 0 for r in rows) / len(rows)
    out = sum(r.get("output_tokens") or 0 for r in rows) / len(rows)
    pi, po = PRICES[model]
    per_call = (inp * pi + out * po) / 1e6
    return {"mean_input": round(inp), "mean_output": round(out),
            "usd": round(per_call * n_calls * (0.5 if batch else 1.0), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calls", type=int, default=2600, help="200 questions x 13 conditions")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.pilot) if l.strip()]
    print(json.dumps(estimate(rows, a.model, a.calls), indent=1))


if __name__ == "__main__":
    main()
