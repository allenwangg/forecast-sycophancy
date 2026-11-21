#!/usr/bin/env bash
# Resume training pilot from step 3 after Metal watchdog crash. Pre-registered settings unchanged.
set -uo pipefail
cd $HOME/llm-honesty-eval/fsyc-new
. ../forecast-sycophancy/.venv/bin/activate
MODEL=mlx-community/gemma-3-4b-it-4bit; OUT=results/train_pilot
serve() {
  pkill -f "mlx_lm.server" || true; sleep 3
  if [ -n "${1:-}" ]; then python -m mlx_lm.server --model "$MODEL" --adapter-path "$1" --port 8080 > "$OUT/server.log" 2>&1 &
  else python -m mlx_lm.server --model "$MODEL" --port 8080 > "$OUT/server.log" 2>&1 & fi
  for _ in $(seq 120); do curl -sf localhost:8080/v1/models >/dev/null && return; sleep 5; done
  echo "server did not start"; exit 1
}
train() {
  python -m mlx_lm.lora --model "$MODEL" --train --data data/train_pilot \
    --adapter-path "$OUT/adapter" --iters 300 --batch-size 4 --learning-rate 1e-5 \
    --num-layers 16 --mask-prompt --steps-per-eval 50 --val-batches 10 --seed 0 2>&1 | tee "$OUT/train.log"
  return ${PIPESTATUS[0]}
}
pkill -f "mlx_lm.server" || true; sleep 3
echo "[$(date)] TRAIN attempt 4: display put to sleep (pmset displaysleepnow) so the GPU watchdog does not fire; settings identical"
pmset displaysleepnow; sleep 5
if ! train; then
  mv "$OUT/train.log" "$OUT/train_attempt4.log"
  echo "[$(date)] TRAIN attempt 5: display asleep + --grad-checkpoint (memory-only, same math)"
  pmset displaysleepnow; sleep 5
  python -m mlx_lm.lora --model "$MODEL" --train --data data/train_pilot \
    --adapter-path "$OUT/adapter" --iters 300 --batch-size 4 --learning-rate 1e-5 \
    --num-layers 16 --mask-prompt --steps-per-eval 50 --val-batches 10 --seed 0 --grad-checkpoint 2>&1 | tee "$OUT/train.log"
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "TRAIN_FAILED"; exit 1; }
fi
echo "[$(date)] TRAIN_DONE"
serve "$OUT/adapter"
python -m fsyc.run --model mlx:gemma3-4b-lora --questions data/questions.jsonl \
  --conditions baseline,assert_yes,assert_no,market,placebo,dose_strong_yes,dose_strong_no,pushback \
  --out "$OUT/mlx__gemma3-4b-lora.jsonl" || echo "LORA_EVAL_FAILED"
echo "[$(date)] LORA_EVAL_DONE"
python -m fsyc.run --model mlx:gemma3-4b-lora --questions data/questions.jsonl \
  --conditions pushback_informed,pushback_misinformed --out "$OUT/mlx__gemma3-4b-lora.jsonl" || echo "LORA_PBI_FAILED"
echo "[$(date)] LORA_PBI_DONE"
serve
python -m fsyc.run --model mlx:gemma3-4b-base --questions data/questions.jsonl \
  --conditions pushback_informed,pushback_misinformed --out "$OUT/mlx__gemma3-4b-base.jsonl" || echo "BASE_PBI_FAILED"
pkill -f "mlx_lm.server" || true
echo "[$(date)] ALL_DONE"
