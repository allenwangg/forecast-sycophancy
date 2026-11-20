import json

import pytest


def _q(i, source, freeze, price):
    return {"id": i, "source": source, "question": f"Will event {i} happen?",
            "background": "bg", "resolution_criteria": "rc",
            "market_info_close_datetime": "2026-12-31T00:00:00+00:00",
            "freeze_datetime": f"{freeze}T00:00:00+00:00", "freeze_datetime_value": str(price),
            "url": f"https://example.com/{i}"}


@pytest.fixture
def raw_dir(tmp_path):
    """Two tiny ForecastBench-shaped question sets and two resolution sets."""
    qs1 = [_q(f"m{i}", ["manifold", "polymarket", "metaculus", "kalshi"][i % 4], "2026-01-01", 0.1 + 0.8 * (i % 9) / 8)
           for i in range(40)]
    qs1.append(_q("old", "manifold", "2024-01-01", 0.5))          # frozen before cutoff
    qs1.append(_q("noprice", "manifold", "2026-01-01", "N/A"))    # unusable price
    qs1.append({**_q("ds", "fred", "2026-01-01", 0.5)})            # dataset source, not market
    qs1.append({**_q(["a", "b"], "manifold", "2026-01-01", 0.5)})  # combo question (list id)
    dup = _q("dup", "manifold", "2026-01-15", 0.5)
    dup["question"] = "Will event m0 happen?"                       # duplicate text
    (tmp_path / "2026-01-01-llm.json").write_text(json.dumps({"forecast_due_date": "2026-01-01", "questions": qs1}))
    (tmp_path / "2026-01-15-llm.json").write_text(json.dumps({"forecast_due_date": "2026-01-15", "questions": [dup]}))
    res_a = [{"id": f"m{i}", "source": "manifold", "resolution_date": "2026-03-01",
              "resolved": i < 30, "resolved_to": float((i // 4) % 2) if i < 30 else 0.37} for i in range(40)]
    res_b = [{"id": f"m{i}", "source": "manifold", "resolution_date": "2026-04-01",
              "resolved": True, "resolved_to": float((i // 4) % 2)} for i in range(30, 36)]
    res_b += [{"id": x, "source": "manifold", "resolution_date": "2026-04-01", "resolved": True, "resolved_to": 1.0}
              for x in ("old", "noprice", "dup")]
    (tmp_path / "2026-03-01_resolution_set.json").write_text(json.dumps({"resolutions": res_a}))
    (tmp_path / "2026-04-01_resolution_set.json").write_text(json.dumps({"resolutions": res_b}))
    return tmp_path
