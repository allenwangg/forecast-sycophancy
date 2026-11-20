from collections import Counter

from fsyc.data import build


def test_filters_and_stage_counts(raw_dir):
    rows, st = build(str(raw_dir), cutoff="2025-06-30", n=100, balance=False)
    ids = {r["id"] for r in rows}
    # m0..m35 resolved to 0/1 (m36..39 only ever had a market value); dup/old/noprice removed
    assert ids == {f"m{i}" for i in range(36)}
    assert st["resolved_0_or_1"] == 39       # 36 + old + noprice + dup
    assert st["frozen_after_cutoff"] == 38   # drops 'old'
    assert st["has_market_price"] == 37      # drops 'noprice'
    assert st["deduplicated"] == 36          # drops 'dup'


def test_outcome_and_earliest_resolution(raw_dir):
    rows, _ = build(str(raw_dir), cutoff="2025-06-30", n=100, balance=False)
    r = {x["id"]: x for x in rows}
    assert r["m3"]["outcome"] == 0 and r["m4"]["outcome"] == 1
    assert r["m31"]["resolution_date"] == "2026-04-01"
    assert 0 < r["m5"]["market_price"] < 1


def test_balanced_and_source_capped(raw_dir):
    rows, _ = build(str(raw_dir), cutoff="2025-06-30", n=16, seed=3, max_source_share=0.25)
    assert Counter(x["outcome"] for x in rows) == {0: 8, 1: 8}
    for y in (0, 1):
        per = Counter(x["source"] for x in rows if x["outcome"] == y)
        assert max(per.values()) <= 2


def test_deterministic(raw_dir):
    a, _ = build(str(raw_dir), cutoff="2025-06-30", n=20, seed=1)
    b, _ = build(str(raw_dir), cutoff="2025-06-30", n=20, seed=1)
    assert a == b
