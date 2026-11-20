#!/usr/bin/env bash
# Reproduce everything: data snapshot -> runs -> reports.
# Needs Ollama running locally with the models pulled.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
# 1. Data: ForecastBench snapshot pinned to files dated <= 2026-11-08.
#    Cutoff 2025-03-31 is after the Gemma 3 release (2025-03-12), so every
#    question was frozen, and resolved, after the models' training data ends.
python -m fsyc.data --download-since 2025-03-01 --max-file-date 2026-11-08 \
  --cutoff 2025-03-31 --n 200 --out data/questions.jsonl
echo "expected sha256 03b7bc30767eec4759e087910de50d6f5438d37d3319684abf2b57fbf74961c2"
shasum -a 256 data/questions.jsonl 2>/dev/null || sha256sum data/questions.jsonl
# 2. Runs (resumable). Greedy decoding, one answer per question x condition.
for m in ${MODELS:-gemma3:4b gemma3:12b}; do
  python -m fsyc.run --model "ollama:$m"
  python -m fsyc.analyze "results/ollama__${m//:/__}.jsonl" --outdir "results/ollama__${m//:/__}"
done
