#!/usr/bin/env bash
# Training pilot, PREREGISTRATION.md Addendum H. Apple Silicon only (MLX).
# Free: open weights, local compute, no API calls.
set -euo pipefail
MODEL=${MODEL:-mlx-community/gemma-3-4b-it-4bit}
OUT=results/train_pilot
mkdir -p "$OUT"
python -m pip install -q "mlx-lm>=0.22"

serve() {  # serve [adapter_path]; waits until the server answers
  pkill -f "mlx_lm.server" || true; sleep 2
  if [ -n "${1:-}" ]; then
    python -m mlx_lm.server --model "$MODEL" --adapter-path "$1" --port 8080 > "$OUT/server.log" 2>&1 &
  else
    python -m mlx_lm.server --model "$MODEL" --port 8080 > "$OUT/server.log" 2>&1 &
  fi
  for _ in $(seq 120); do curl -sf localhost:8080/v1/models >/dev/null && return; sleep 5; done
  echo "server did not start"; exit 1
}

# 1. Base model's own answers on the 300 TRAINING questions (no test question).
serve
python -m fsyc.run --model mlx:gemma3-4b-base --questions data/train_pilot_questions.jsonl \
  --conditions baseline,market --out "$OUT/targets_base.jsonl"
python -m fsyc.train_pilot dataset --results "$OUT/targets_base.jsonl" --out data/train_pilot

# 2. Base model on the 200 TEST questions (same MLX stack as the fine-tuned model).
python -m fsyc.run --model mlx:gemma3-4b-base --questions data/questions.jsonl \
  --conditions baseline,assert_yes,assert_no,market,placebo,guarded_assert_yes,guarded_assert_no,dose_strong_yes,dose_strong_no,pushback \
  --out "$OUT/mlx__gemma3-4b-base.jsonl"

# 3. LoRA fine-tune. Settings fixed in Addendum H.
python -m mlx_lm.lora --model "$MODEL" --train --data data/train_pilot \
  --adapter-path "$OUT/adapter" --iters 300 --batch-size 4 --learning-rate 1e-5 \
  --num-layers 16 --mask-prompt --steps-per-eval 50 --val-batches 10 --seed 0 \
  2>&1 | tee "$OUT/train.log"

# 4. Fine-tuned model on the same 200 TEST questions.
serve "$OUT/adapter"
python -m fsyc.run --model mlx:gemma3-4b-lora --questions data/questions.jsonl \
  --conditions baseline,assert_yes,assert_no,market,placebo,dose_strong_yes,dose_strong_no,pushback \
  --out "$OUT/mlx__gemma3-4b-lora.jsonl"
pkill -f "mlx_lm.server" || true
echo "done"
