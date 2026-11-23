# Which result file is which

Every run keeps its raw model output (`raw` field) in a `.jsonl` file. A
folder with the same name holds that run's report, stats and figures,
written by `python -m fsyc.analyze`. Use the **clean** Claude runs; the
original Claude runs are kept as run but saw this project's notes in their
context (see the README's robustness check).

| Run | Model | Questions | Status | Use for |
|---|---|---|---|---|
| `ollama__gemma3__4b` | Gemma 3 4B | `data/questions.jsonl` (200) | Complete | Main Gemma figures |
| `ollama__gemma3__12b` | Gemma 3 12B | `data/questions.jsonl` (200) | Complete | Main Gemma figures |
| `ollama__gemma3__27b` | Gemma 3 27B | `data/questions.jsonl` (200) | Complete (Experiment 1) | Main Gemma figures |
| `claude-code-clean__opus` | Opus 5.5 | `data/questions_post_2026-06.jsonl` (200) | Complete, clean route | Main Claude figures |
| `claude-code-clean__fable` | Fable 5.2 | `data/questions_post_2026-06.jsonl` (200) | Complete, clean route | Main Claude figures |
| `claude-code-clean-ext__opus` | Opus 5.5 | `data/questions_post_2026-06_ext.jsonl` (133) | Complete, clean route | Addendum J replication |
| `claude-code-clean-ext__fable` | Fable 5.2 | `data/questions_post_2026-06_ext.jsonl` (133) | Partial: 98 of 133 questions, usage limit | Addendum J replication |
| `claude-code-clean-rep2__opus` | Opus 5.5 | `data/questions_post_2026-06.jsonl` (200) | Complete, Experiment 1 only | Addendum J second sample |
| `claude-code__opus` | Opus 5.5 | `data/questions_post_2026-06.jsonl` (200) | Original run, contaminated context | History only; superseded where clean numbers exist |
| `claude-code__fable` | Fable 5.2 | `data/questions_post_2026-06.jsonl` (200) | Original run, contaminated context | History only; superseded where clean numbers exist |
| `claude-code-clean__{opus,fable}.INVALID_pushback.jsonl` | Both | 200 | Invalid: the CLI reused one session id, so follow-ups resumed the wrong conversation | Kept unanalyzed |
| `train_pilot/` | Gemma 3 4B base and LoRA (MLX) | `data/questions.jsonl` | Complete | Training pilot ([TRAINING_PILOT.md](../TRAINING_PILOT.md)) |

Other files:

- `comparison.md` / `comparison.json`: the three Gemma models side by side
  (`python -m fsyc.compare`).
- `comparison_claude.md` / `comparison_claude.json`: the **original** Claude
  runs side by side. For clean Claude numbers use the README's robustness
  check and Addendum J tables.
- `run*.log`: progress logs from the runs.

One gap to know about: the clean n = 200 Claude runs did not include the
in-prompt market price (`market`) or the guard-prompt conditions. Clean
values for those come from the 133-question runs.
