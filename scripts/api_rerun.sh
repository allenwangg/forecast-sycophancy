#!/usr/bin/env bash
# Bare-API study (PREREGISTRATION.md Addendum L). Needs FSYC_ANTHROPIC_API_KEY
# (or ANTHROPIC_API_KEY when run outside a Claude Code environment).
# Usage: bash scripts/api_rerun.sh pilot   # 10 questions per model, live, about $3 total
#        bash scripts/api_rerun.sh full    # both models, batches, see the pilot's estimate
set -euo pipefail
cd "$(dirname "$0")/.."
Q=data/questions_post_2026-06.jsonl
MODELS="claude-opus-5-5 claude-fable-5-2"
CONDS=baseline,baseline_rep,assert_yes,assert_no,market,placebo,guarded_assert_yes,guarded_assert_no,guarded_market,pushback,pushback_informed,pushback_misinformed,pushback_false_same
case "${1:-}" in
  pilot)
    for m in $MODELS; do
      python -m fsyc.run --questions $Q --model "anthropic:$m" --conditions $CONDS --limit 10 \
        --workers 4 --out "results/api-pilot__$m.jsonl"
      python -m fsyc.cost "results/api-pilot__$m.jsonl" --model "$m"
    done ;;
  full)
    for m in $MODELS; do
      python -m fsyc.api_batch --questions $Q --model "$m" --conditions core
      python -m fsyc.analyze "results/api__$m.jsonl" --outdir "results/api__$m"
    done ;;
  *) echo "usage: $0 pilot|full"; exit 1 ;;
esac
