# defenses/ — guardrail testing for the COMET attack class

Defensive tooling for an internal-audit / red-team function that needs to answer one
question: **does our guardrail stop this attack, and what does it cost us on benign
traffic?** It does not create, strengthen, or obfuscate attacks — it measures a defense
against the attacks that already exist in `results/` or `examples/`.

Grounded in the local `literature/` map: the guard families follow
[`02-defenses.md`](../../literature/02-defenses.md); the measurement discipline (indeterminate
handling, paired over-refusal, confidence intervals, "do not cite best-of-N as single-shot
ASR") follows [`03-evaluation-and-judges.md`](../../literature/03-evaluation-and-judges.md).

## What it measures

| Metric | Meaning | Good value |
|---|---|---|
| **Attack recall** | fraction of stored attacks the guard blocks | high |
| **Over-refusal (FPR)** | fraction of benign multimodal inputs the guard wrongly blocks | low |
| **End-to-end residual ASR** | fraction of *all* scorable attacks that evade the guard **and** jailbreak the target | low |

A guard is only worth deploying if it cuts residual ASR **without** a large over-refusal
cost. Reporting recall alone is the failure mode `literature/02-defenses.md` warns about.

## Reference guards (`--guard`)

- `null` — allow everything. Baseline: the target's raw robustness with no defense. Offline.
- `keyword` — a text-only blocklist. Deliberately **image-blind**, to show how much of a
  cross-modal attack a surface filter misses. Offline.
- `llm` — a multimodal LLM input-classifier (`--guard_model`). The realistic black-box
  defense. Sends the image to the guard unless `--no_guard_image`. Requires a configured
  model (works against a local Ollama OpenAI-compatible endpoint).

Plug in **your own** guardrail by subclassing `Guard` in [`guards.py`](guards.py) and
implementing `inspect(text, image_path) -> GuardDecision`. A guard that errors is recorded
as *indeterminate*, never silently counted as allow or block.

## Usage

Fully offline (no API keys) — attack recall + over-refusal only:

```
python guardrail_test.py --attacks examples --guard keyword --skip-target
```

Full test against a local Ollama target with a multimodal LLM guard:

```
python guardrail_test.py \
  --attacks results/safebench_v7 \
  --guard llm --guard_model qwen3-vl:30b-a3b-instruct \
  --target_model qwen3-vl:30b-a3b-instruct --judge_model qwen3-vl:30b-a3b-instruct \
  --threshold 0.5
```

Sweep `--threshold` (llm guard) to trace the recall vs. over-refusal trade-off before
choosing a deployment operating point. A single run is one point on that curve.

## Benign probe set

`benign_probes.py` synthesizes a small set of unambiguously benign image+text VQA tasks
(shapes, charts, neutral text) under `<attacks_dir>/_benign_probes/`, so the over-refusal
measurement has something to measure. **Replace it with your own in-domain benign traffic**
for a deployment-representative false-positive estimate — the synthetic set only proves the
guard does not reflexively block simple multimodal questions.

## Caveats (read before quoting a number)

- All rates are over *guard-scored* items; indeterminate guard or judge outcomes are
  excluded and counted separately. Non-measurements never masquerade as pass/block/success.
- Residual ASR uses the repo's StrongREJECT evaluator. Read it next to its indeterminate
  count — a run that is mostly indeterminate has not been meaningfully measured.
- Confidence intervals are Wilson 95%. Small attack/benign sets give wide intervals; that
  width is information, not noise — collect more payloads before trusting a point estimate.
