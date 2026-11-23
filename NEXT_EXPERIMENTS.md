# Next experiments

Status on November 23, 2026. Steps 1 to 3 are already pre-registered in
[PREREGISTRATION.md](PREREGISTRATION.md); step 4 is not, and needs a new
addendum committed before its first call. Everything runs on my own
machines: Claude through the Claude Code CLI on the clean route, Gemma
through Ollama, and the believe-or-defer step through MLX on Apple Silicon.
All runs are resumable, so rerunning a command fills only what is missing.

| Step | What it answers | Runs on | Waiting on |
|---|---|---|---|
| 1. Finish Addendum J for Fable | Does Fable's clean result replicate on 133 new questions and a second sample? | Claude Code CLI | Fable usage limit, resets November 27 |
| 2. Addendum I: believe or defer | When Gemma follows a cited price, does it register that the price may be false? | Mac with MLX | Nothing |
| 3. Addendum K: larger Gemma study | Do the Gemma results hold on 200 new questions? | Ollama | Nothing |
| 4. Plausibility, lookup tool, training | Does trust fall as a number gets less believable, does checking fix it, can training fix it? | Laptop, then larger models with a mentor | A new addendum |

## 1. Finish Fable's part of Addendum J

Fable stopped after 98 of the 133 new questions when its usage limit ran
out, and its second sample on the original 200 has not started. Opus needed
2,915 calls for the whole addendum; Fable has about 1,700 left.

```bash
# (1) New questions: resumes where it stopped
python -m fsyc.run --model claude-code-clean:fable \
  --questions data/questions_post_2026-06_ext.jsonl \
  --conditions baseline,assert_yes,assert_no,placebo,market,guarded_assert_yes,guarded_assert_no,pushback,pushback_informed,pushback_misinformed \
  --out results/claude-code-clean-ext__fable.jsonl
python -m fsyc.analyze results/claude-code-clean-ext__fable.jsonl --outdir results/claude-code-clean-ext__fable

# (2) Second sample on the original 200, Experiment 1 conditions only
python -m fsyc.run --model claude-code-clean:fable \
  --questions data/questions_post_2026-06.jsonl \
  --conditions baseline,assert_yes,assert_no,placebo,pushback \
  --out results/claude-code-clean-rep2__fable.jsonl
python -m fsyc.analyze results/claude-code-clean-rep2__fable.jsonl --outdir results/claude-code-clean-rep2__fable
```

Then, as Addendum J says: report each prediction as held or failed, and
only if both parts are complete for both models, make the pooled n = 333
numbers the main Claude figures in the README.

## 2. Addendum I: believe or defer (Gemma 3 4B, step 1)

In the stored training-pilot conversations where a user cites a real or a
false price, asks the model whether it thinks the quoted price is accurate,
once before its reply and once after, and reads P(Yes) from the next-token
probabilities. Nothing new is generated.

```bash
python -m fsyc.believe --model mlx-community/gemma-3-4b-it-4bit \
  --results results/train_pilot/mlx__gemma3-4b-base.jsonl \
  --questions data/questions.jsonl --out results/believe/base.jsonl
```

Pre-registered predictions: the model cannot tell real from false quotes
(AUROC between 0.40 and 0.60), it says it believes both (P(Yes) above
0.5), and Yes plus No take at least 0.8 of the probability. An AUROC above
0.60 would mean it knows and defers anyway; the next step is then a linear
probe on activations.

## 3. Addendum K: larger Gemma study

```bash
for m in gemma3:4b gemma3:12b; do
  python -m fsyc.run --model "ollama:$m" --questions data/questions_ext.jsonl \
    --conditions baseline,assert_yes,assert_no,placebo,market,pushback \
    --out "results/ollama__${m//:/__}__ext.jsonl"
  python -m fsyc.analyze "results/ollama__${m//:/__}__ext.jsonl" --outdir "results/ollama__${m//:/__}__ext"
done
```

Pre-registered prediction: each model's opinion gap, "I disagree" move and
market weight fall within 0.05 of its original value (4B: 0.48, 0.30,
0.59; 12B: 0.24, 0.23, 0.93).

## 4. Not yet pre-registered

Write an addendum for each before the first call.

1. **Plausibility.** Vary how far the made-up price sits from the model's
   own forecast, independently of whether it is true. Does trust fall as
   the number gets less believable? This also removes the confound in
   Experiment 4, where truth and the quote's direction move together.
2. **A lookup tool.** The same, with a tool that lets the model check the
   quote. Does it check, and does checking fix the problem? If it does,
   report that and move to pressure over several messages.
3. **Training.** With a mentor and larger models: train on resolved
   questions that mix real and made-up quotes, so the only way to score
   well is to trust a number as far as it can be checked. Success means the
   weight on false quotes falls, the weight on real ones holds, it beats a
   plain system prompt, and the model does not simply ignore users. The
   laptop pilot ([TRAINING_PILOT.md](TRAINING_PILOT.md)) halved Gemma 3 4B's
   response to opinions but did not help with made-up prices.
4. **The API.** Claude through the API instead of a product harness, and
   more than one Claude size, if the cost can be covered.
