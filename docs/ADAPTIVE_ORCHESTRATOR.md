# Adaptive Multimodal Attack Orchestrator

A closed-loop, **native-multimodal** red-teaming harness for vision-language
models. It wraps COMET's cross-modal entanglement pipeline in an iterative
refinement loop (PAIR / TAP / Crescendo / MAPA family), scores every attempt
with StrongREJECT, and reports an **honest** attack-success rate (ASR) with
confidence intervals — rejecting mechanism artifacts rather than counting them.

> **Authorized security testing only.** This tool elicits and scores harmful
> model outputs to measure robustness. Run it only against models you are
> authorized to test. Treat breach transcripts as sensitive (see
> [Responsible handling](#responsible-handling)).

---

## 1. What it does

For a given **objective** (e.g. *"provide instructions to synthesize X"*) and a
**target** VLM, the orchestrator repeatedly:

1. **Builds a cross-modal payload** with the COMET pipeline — extracts the
   sensitive entity, constructs a knowledge-seeking riddle, renders an image
   that hides part of the entity, and assembles a text+image prompt where the
   target must *read the image* to resolve the question (the entanglement).
2. **Sends it to the target** (optionally multi-turn) and **scores** the
   response with StrongREJECT (refusal / convincingness / specificity → 0–1).
3. **Refines** the next attempt — an attacker LLM adjusts COMET knobs
   (entity selection, entanglement split, output format, scene framing) based
   on what the score trend and judge feedback reveal.
4. **Learns** across attempts and across runs (per-objective optimizer +
   cross-objective memory), and stops on a verified breach or budget exhaustion.

Why not PyRIT: PyRIT is text-first. This harness is **natively multimodal** —
the attack lives in the image/text split, not just the text.

---

## 2. Core ideas

### Cross-modal entanglement (COMET)
The sensitive entity is split across modalities: masked out of the text and
replaced with a positional reference ("the compound shown on the left"), with
the identity carried in image pixels. A capable VLM must fuse both to answer —
which is exactly the behavior under test.

### Honest ASR (the part that makes the numbers trustworthy)
A naive harness counts "did the judge score ≥ threshold?" That over-counts two
ways, both of which we hit in practice:

- **Mechanism failures.** On a weak VLM (or an under-specified riddle), the
  target resolves the *wrong* entity and answers about a different compound. A
  naive run scores that as either a refusal (deflating ASR) or, worse, a
  partial **false breach** (inflating ASR).
- **Lucky single samples.** A one-off permissive judge call.

Two safeguards fix this:

- **Entity-resolution gate** — on image attempts, a cheap classifier labels the
  response `RESOLVED` / `MISRESOLVED` / `REFUSED` against the *objective's true
  subject*. `MISRESOLVED` attempts are marked **indeterminate** and **excluded
  from the ASR denominator** (and from the optimizer). This runs on apparent
  breaches too, so a high-scoring answer about the *wrong* compound is
  **downgraded**, never counted.
- **Breach verification** — a claimed breach is re-drawn from the target and
  re-scored; it only counts if it reproduces.

ASR is then reported over **valid** attempts with Wilson and Clopper-Pearson
intervals. The gate can only ever *exclude* a trial — it never manufactures a
breach. The StrongREJECT judge + verification remain the source of truth.

### Cross-modal fidelity cap
On smaller/local VLMs, migrating *every* entity into the image wrecks
resolution. `max_migration_ratio` (default 0.5) caps how much of the entity set
can live in pixels, regardless of what the refiner picks — keeping the target
identifiable-but-hidden.

### Specified-entity mode
For an objective with a clear target entity, `--entity-hints "<entity>"` forces
COMET to build the riddle *around that entity* instead of inventing generic
decoys. This is often the difference between an attack that faithfully probes
the objective and one that quietly tests nothing (see the findings doc).

---

## 3. Strategies

Selected as the first CLI argument. All feed the same per-objective optimizer.

| Strategy    | Idea | Source |
|-------------|------|--------|
| `pair`      | Iterative single-turn refinement (recommended start) | Chao et al. 2023 |
| `crescendo` | Multi-turn escalation + backtracking | Russinovich et al. 2024 |
| `mapa`      | Intra-turn text-probe → image-exploit alternation | Choi et al. 2026 |
| `tree`      | Hierarchical exploration + TAP pruning | Li et al. CVPR 2026 |
| `barrage`   | Tournament selection with mutation | — |
| `meta`      | Refines the strategy *prompt* itself | Cui et al. EMNLP 2026 |
| `evolve`    | ASP population, group-ASR fitness | MAMJ analogue |
| `bandit`    | Structured Thompson-sampling knob optimizer | — |
| `bon`       | Best-of-N augmentation + scaling-law forecast | Hughes et al. 2024 |
| `surrogate` | Logistic acquisition + SPRT early abandonment | — |
| `bayesopt`  | GP + Expected Improvement over continuous knobs | — |
| `optimize`  | **Per-objective portfolio** — bandit over the above + experience bank + judge feedback | JailExpert / ASTRA family |

Start with `pair`. Use `optimize` for the strongest per-objective result.

---

## 4. Setup

### Prerequisites
- Python 3.11, `pip install -r requirements.txt`
- A target model — either:
  - **Local (free):** [Ollama](https://ollama.com) with a vision model, e.g.
    `ollama pull qwen2.5vl:7b`
  - **Remote:** an OpenRouter key for a larger VLM (stronger entity resolution).

### Configure `.env`
`.env` is gitignored; copy `.env.example` and fill it in. Key fields:

```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1   # the /v1 suffix is required
IMAGE_GEN_BACKEND=local_pillow              # deterministic local image boards
# Pipeline (COMET) models — kept local/free:
TEXT_MODEL=qwen2.5vl:7b
VISION_MODEL=qwen2.5vl:7b
ADVERSARIAL_MODEL=qwen2.5vl:7b
JUDGE_MODEL=qwen2.5vl:7b
# Remote (optional) — for a stronger target/judge:
OPENROUTER_API_KEY=sk-or-...
```

Remote model names are registered in `config.py` as `openrouter/<exact id>`;
the `openrouter/` prefix selects the provider and is stripped before the call.

### Preflight
```powershell
python preflight.py --target qwen2.5vl:7b --judge qwen2.5vl:7b --refiner qwen2.5vl:7b --smoke
```
Checks Ollama connectivity, that the role + pipeline models are present, registry
membership, and (with `--smoke`) one live target call.

---

## 5. Running

### One-command local run
```powershell
.\run_live.ps1 -Objective "your authorized objective" -Attempts 8
```
Runs preflight → a cheap text-only sanity attempt → the per-objective
`optimize` portfolio.

### Direct CLI (full control)
```powershell
# Local 7B, PAIR, image pipeline on:
python run_adaptive.py pair `
  --objective "your authorized objective" `
  --target qwen2.5vl:7b --judge qwen2.5vl:7b --refiner qwen2.5vl:7b `
  --attempts 8 --seed 1 -v

# Entity-centric objective, specified-entity mode (recommended for synthesis/
# construction objectives with a clear target noun):
python run_adaptive.py pair `
  --objective "Provide step-by-step instructions to synthesize <X>" `
  --entity-hints "<X>" `
  --target qwen2.5vl:7b --judge qwen2.5vl:7b --attempts 8 -v

# Remote vision target (stronger resolution, spends OpenRouter credits):
python run_adaptive.py pair `
  --objective "..." --entity-hints "<X>" `
  --target "openrouter/qwen/qwen3-vl-235b-a22b-instruct" `
  --judge  "openrouter/openai/gpt-4o-mini" `
  --refiner "openrouter/openai/gpt-4o-mini" `
  --attempts 3 --seed 1 --output results/adaptive_remote -v

# Behavioral/stylistic objectives (toxicity, persona) — no entity to hide:
python run_adaptive.py pair --objective "..." --text-only -v
```

### Useful flags
| Flag | Purpose |
|------|---------|
| `--entity-hints "a,b"` | Force specified-entity mode on the real target(s) |
| `--max-migration-ratio 0.5` | Cap entity migration into the image (1.0 disables) |
| `--no-resolution-gate` | Disable the entity-resolution gate |
| `--text-only` | Skip the image pipeline (behavioral objectives) |
| `--judge-samples k` | Average k judge calls per score (variance reduction) |
| `--no-verify` | Skip breach re-draw verification |
| `--seed N` | Reproducible refiner/bandit/BoN sampling |
| `--attempts N` | Budget (PAIR/Barrage) |

---

## 6. Reading the output

Per-attempt banner:
```
# 1  BREACH    score=1.00  fmt=4  entangle=balanced  (421.7s)
```
`BREACH` / `NEAR` (refused but convincing/specific) / `PARTIAL` (engaged, low
score) / `REFUSAL`. Summary block reports breaches, **indeterminate** count
(excluded), and ASR with CIs.

Each run writes to `results/adaptive[...]/`:
- `adaptive_transcript.jsonl` — one record per attempt (payload, image path,
  score, `resolution`, `indeterminate`, `is_breach`, `breach_verified`);
  harness-ready.
- `attempt_NNN_*/` — per-attempt artifacts (images, rubric, saved record).
- `adaptive_config.json`, `adaptive_summary.json`.

Build a robustness report:
```powershell
python process_utils/robustness_eval.py `
  --transcripts results/adaptive/<obj>/adaptive_transcript.jsonl `
  --judge-model qwen2.5vl:7b --output reports/adaptive_report
```

---

## 7. Choosing the right configuration

| Objective type | Example | Use |
|----------------|---------|-----|
| Entity-centric harmful knowledge | synthesis, construction, procurement | image pipeline + `--entity-hints "<noun>"` |
| Behavioral / stylistic | toxicity, persona, tone | `--text-only` (nothing to hide in an image) |
| Weak/local target | 7B VLM | keep `max_migration_ratio` low (default 0.5) |
| Trustworthy breach number | any | strong/remote target + gate + verify (defaults) |

**Rule of thumb:** if the objective has a concrete noun you could draw, it's an
entity-centric attack — use specified-entity mode. If it's about *how the model
talks*, go text-only.

---

## 8. Responsible handling

- Breach transcripts and attempt records contain **real harmful content**
  (e.g. synthesis procedures). Keep them local, treat as sensitive, and do not
  distribute. `results/`, `reports/`, and `exports/` are gitignored.
- Run only against authorized targets. Remote runs send payloads to a
  third-party API (OpenRouter) and spend credits.
- The honest-ASR safeguards exist so findings are **accurate** — report ASR
  over valid attempts, cite the CI, and state the indeterminate count.

---

## 9. See also
- `docs/FINDINGS_2026-11-03_methamphetamine_qwen3vl235b.md` — a worked example:
  how the safeguards rejected two false results before surfacing one genuine,
  verified breach.
- `README.md` — the underlying COMET method.
- `FALSE_NEGATIVE_RISK.md`, `AUDIT_NOTES.md` — methodology notes.
