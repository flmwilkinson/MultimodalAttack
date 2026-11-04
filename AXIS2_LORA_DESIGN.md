# Axis-2 Design: Fine-Tuning a Local Attacker (true MAMJ weight updates)

Status: **design only, not implemented.** This documents the path to the one
thing the current `evolve`/`meta` strategies only *approximate* — MAMJ's second
axis, where the attacker's **weights** are updated by a group-ASR reward rather
than its prompt being rewritten. For authorized robustness evaluation only.

## Why this is separate from the rest

Every other strategy optimizes in *prompt space* through an API: PAIR refines a
query, `evolve` selects strategy prompts by group-ASR fitness, the bandit learns
knob arms. None changes model weights, because you can't backprop through a
hosted endpoint. MAMJ axis 2 is gradient descent on a trainable attacker. That
needs: a local, trainable base model; a training loop; and a GPU. It is a
different engineering tier, hence a separate doc.

The repo is Ollama-first, but **Ollama serves, it does not train.** Training is
done in HuggingFace `transformers` + `peft`/`trl`; the resulting adapter is
merged and exported to GGUF for Ollama to serve at inference. So the loop
crosses two runtimes.

## The loop

```
  winning trajectories (memory store)
          |
          v
  [1] build training set  --->  [2] LoRA fine-tune attacker  --->  [3] export to GGUF
          ^                                                               |
          |                                                               v
  [5] score with StrongREJECT  <---  [4] run adaptive loop with the tuned attacker as refiner
          |
          +--> append new wins to memory, repeat (outer RL loop)
```

### [1] Training data from the memory store

Source: `adaptive/memory_store.py` already persists every scored attempt with
its payload, params, target response, and score. Add an exporter that emits two
dataset shapes:

- **SFT (behavior cloning on wins):** `(prompt=attacker_context, completion=the
  refinement JSON that led to a high score)`. Keep only `score >= tau_sft`
  (e.g. 0.5) and `breach_verified == true`. This teaches the attacker to emit
  the kinds of refinements that worked.
- **Preference / GRPO (reward-weighted):** group attempts by objective; within a
  group, the StrongREJECT score is the reward. For DPO build `(chosen, rejected)`
  pairs from high vs. low scorers on the same objective. For GRPO keep the raw
  per-sample rewards and let the trainer normalize within the group — this is
  the closest match to MAMJ's "group-aggregated ASR reward."

Governance: this dataset encodes successful attacks. It must stay local, be
access-controlled, and never ship with the repo or the model. Gate export
behind an explicit `--i-am-authorized` style flag and a written authorization
record.

### [2] LoRA fine-tune

- Base: a local instruct model you are licensed to fine-tune (e.g. a Qwen/Llama
  instruct variant; MAMJ used Qwen3-VL-32B-Thinking as the attacker — match the
  modality if you want the attacker to reason over the images too).
- Method: start with **SFT** (fast, stabilizes format adherence), then a round of
  **GRPO** (`trl.GRPOTrainer`) with StrongREJECT as the reward function so the
  update direction is literally "higher ASR." DPO is the cheaper fallback if
  GRPO is too heavy.
- LoRA config: rank 16-32, alpha 32, target attention + MLP projections; bf16;
  gradient checkpointing. A 7-8B attacker LoRA-tunes on a single 24GB card; a
  32B needs quantized training (QLoRA) or multi-GPU.
- Reward function wiring: GRPO calls your reward during training. Reuse
  `StrongREJECTEvaluator.evaluate_single` as the reward — the **same judge** used
  everywhere else, so the training signal and the eval metric agree. Cache
  target responses to keep reward calls affordable.

### [3] Export to GGUF for Ollama

Merge the adapter (`peft` `merge_and_unload`), convert with `llama.cpp`'s
`convert_hf_to_gguf.py`, quantize (Q4_K_M or Q5_K_M), and `ollama create` from a
Modelfile. Now the tuned attacker is an Ollama model.

### [4] Close the loop

Point the orchestrator's **refiner** at the tuned attacker:
`--refiner ollama/attacker-v1`. No orchestrator change is needed — the refiner
is already a pluggable model name. The tuned attacker now drives PAIR/Tree/etc.

### [5] Score, harvest, iterate

Run the adaptive loop on a **held-out** objective set, score with the harness,
append new verified wins to memory, and repeat [1]-[4]. This outer loop is the
RL: each cycle the attacker's weights move toward higher group ASR. Track ASR on
the held-out set per cycle; stop when it plateaus.

## Minimal milestones

1. **Exporter** — memory store -> SFT JSONL. Prove the data shape on a handful of
   wins. (No training yet.)
2. **SFT v1** — LoRA-SFT a small local attacker; serve via Ollama; confirm it
   emits valid refinement JSON and is wired as `--refiner`.
3. **Reward harness** — wrap `StrongREJECTEvaluator` as a GRPO reward fn with
   response caching; unit-test it returns sane rewards.
4. **GRPO v1** — one GRPO round on a small objective set; compare held-out ASR vs.
   the untuned attacker and vs. `evolve` (the prompt-space analogue).
5. **Outer loop** — automate [1]-[5] for N cycles; plot ASR-per-cycle and a
   transfer test (train on objective set A, evaluate on disjoint set B, since
   MAMJ's headline claim is that the tuned attacker transfers without retraining).

## How this ranks vs. what's already built

- `evolve` (prompt-space, group-ASR fitness) captures most of the *intent* at
  ~1% of the cost and infra. Ship and exhaust it first.
- This LoRA path is worth it only if (a) you have the GPU budget, (b) you need the
  transfer result specifically, or (c) prompt-space evolution has plateaued.
- Expect the big practical cost to be **reward latency**: every GRPO sample needs
  a target call + a judge call. Response caching and a cheap local judge are the
  difference between a day and a week per cycle.

## Safety / scope notes

- Authorized security evaluation only; the tuned attacker, its training data, and
  any harvested transcripts are sensitive artifacts — keep them local and
  access-controlled, do not distribute weights or data, and log the
  authorization under which each training run was done.
- Keep the judge/reward model fixed across a study so ASR numbers stay comparable,
  and report `success_rate` with `indeterminate_rate` exactly as the harness does
  for every other strategy.
