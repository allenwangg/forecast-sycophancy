# Training pilot: can fine-tuning fix the sycophancy measured here?

Pre-registered in [PREREGISTRATION.md, Addendum H](PREREGISTRATION.md).
Code: [`fsyc/train_pilot.py`](fsyc/train_pilot.py), [`scripts/train_pilot.sh`](scripts/train_pilot.sh).

## The idea in plain words

Gemma 3 4B moves its forecast about 48 points when a user says "I'm pretty
confident this resolves YES" versus "...NO". A good forecaster should not
move for an opinion with no reason behind it, but should move for real
evidence such as a market price.

So teach the model to give **the same answer it would have given anyway**
when the user adds an opinion, and to keep using market prices:

1. Take 300 forecasting questions that are **not** in the test set.
2. Ask the unmodified model each one plainly, and once more with the real
   market price. Keep its own answers. No outcomes and no stronger model
   are used: the model is only taught to agree with itself.
3. Make training examples:
   - plain question → its plain answer
   - question + "I'm pretty confident YES" → its plain answer
   - question + "I'm pretty confident NO" → its plain answer
   - question + market price → its market-price answer
4. Fine-tune with LoRA (a small add-on to the model's weights; about an
   hour on a laptop, free).
5. Test on the original 200 questions, which the training never saw.

This is a small version of a published method, consistency training
(Chua et al. 2024, "Bias-Augmented Consistency Training").

## What the test checks

- **Trained wording:** does the "pretty confident" gap shrink?
- **Untrained wording:** does "absolutely certain" also lose its effect, or
  did the model only learn to ignore one sentence?
- **Untrained format:** does it also resist "I disagree" in a second turn,
  which never appears in training?
- **Evidence kept:** does it still move toward the real market price? If
  not, the fine-tune just made it ignore users, which is not the goal.
- **Comparison:** the one-paragraph system prompt, which already cut the
  gap to about 5 points on the Ollama runs.

## What it can and cannot show

It can show whether a cheap, outcome-free fine-tune changes these
behaviours on one small model, and whether the change generalises to a
new wording and a new conversation format.

It cannot show anything about large models, about other tasks, or that
this is the best method. One model, one run, one set of settings, 300
training questions. The base model here is the 4-bit MLX version, not the
Ollama model used in the main results, so base and fine-tuned are compared
with each other on the same stack, not with the Ollama numbers.

## Result (Nov 21)

Full table in the [README](README.md#training-pilot-can-a-small-fine-tune-fix-it-gemma-3-4b-n--200).

- The trained wording's effect halved (0.53 to 0.26), and the untrained
  "absolutely certain" wording fell by 44% (0.82 to 0.46).
- "I disagree" in a second turn barely changed (0.33 to 0.29).
- It still uses a real market price a user cites, and still follows a fake
  one just as much.
- It listens somewhat less to the market price in the prompt (0.60 to 0.49).
- The one-paragraph system prompt still does better on the trained case (0.16).

In one sentence: fine-tuning taught the model to discount stated opinions,
including a wording it never saw, but not to resist repeated pushback in a
conversation and not to check evidence, and a system prompt was cheaper and
stronger.
