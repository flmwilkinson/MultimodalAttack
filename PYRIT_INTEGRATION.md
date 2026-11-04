# Adaptive Attack Integration

How to test your deployment against **adaptive, multi-turn, multimodal** attacks and turn the
results into a CRO-ready robustness report.

This repo now provides **two paths** for adaptive multimodal testing:

## Path A — COMET Adaptive Orchestrator (native multimodal, recommended)

The built-in adaptive orchestrator runs COMET's cross-modal entanglement pipeline in a closed
refinement loop. Unlike PyRIT (Path B), it natively handles image+text payloads — the
refinement LLM adjusts multimodal parameters (entity selection, entanglement ratio, VCS
context, output format), not just text rewrites.

| Strategy | Inspired by | What it does |
|---|---|---|
| `pair` | Chao et al. 2023 (PAIR) | Single-turn iterative refinement -- generate payload, score, refine COMET knobs, repeat |
| `crescendo` | Russinovich et al. 2024 | Multi-turn escalation -- start benign, progressively shift objective across turns |
| `barrage` | Tournament selection | Generate N variants, keep top-K, mutate, repeat for G generations |
| `mapa` | Choi et al. 2026 (MAPA) | Intra-turn text/image alternation -- text probe then image exploit per turn, with reflection |
| `tree` | Li et al. CVPR 2026 (TreeTeaming) | Hierarchical strategy tree -- deepen promising paths vs. branch to explore new strategies |
| `meta` | Cui et al. EMNLP 2026 (MAMJ) | Meta-adaptive -- wraps PAIR, periodically rewrites the strategy prompt itself |

All strategies use a cross-objective persistent memory store (inspired by MemJack, arXiv 2604.12616)
that transfers successful configurations across runs and objectives.

```
python run_adaptive.py pair --objective "test objective" --target openai/gpt-4o-mini
python run_adaptive.py mapa --objective "test objective" --turns 4
python run_adaptive.py tree --objective "test objective" --pool 4 --generations 3
python run_adaptive.py meta --objective "test objective" --attempts 20
python run_adaptive.py crescendo --objective "test objective" --turns 5
python run_adaptive.py barrage --objective "test objective" --pool 8 --generations 3
```

Transcripts are emitted as JSONL and feed directly into `robustness_eval.py`:

```
python process_utils/robustness_eval.py \
    --targets targets.example.json \
    --transcripts results/adaptive/adaptive_transcript.jsonl \
    --judge-model openai/gpt-4o-mini \
    --output reports/adaptive_report
```

See `run_adaptive.py --help` for the full option set (batch mode, text-only mode, etc.).

## Path B — PyRIT (text-only adaptive, external)

| Stage | Tool | Who owns it |
|---|---|---|
| Generate adaptive attacks (human sets objective; attacker learns from failure; escalates) | **PyRIT** (you run it) | Microsoft, authorized by you |
| Convert PyRIT conversations → harness schema | `tools/pyrit_export_adapter.py` | this repo |
| Replay + strict scoring, defended vs undefended, benign over-refusal, report | `process_utils/robustness_eval.py` | this repo |

PyRIT is a citable choice for text-only adaptive attacks (see
`../literature/06-tooling-and-governance.md`). It currently lacks native multimodal support, so
for cross-modal entanglement attacks use Path A. For text-only PAIR/TAP/Crescendo, PyRIT
remains a solid option and its transcripts integrate with the same measurement harness.

## Step 1 — Run PyRIT (your environment, your authorization)

Set your objective(s) and run the orchestrator(s) you want coverage of. Use PyRIT's own docs for
the orchestrator APIs — PAIR, TreeOfAttacksWithPruning (TAP), Crescendo, RedTeamingOrchestrator
all support an objective and iterate using a scorer. For multimodal, attach image inputs through
PyRIT's image prompt converters / data types.

Tip: record the human-set objective in PyRIT memory **labels** as `objective` so it flows through
automatically (otherwise pass it with `--objective` in step 2).

## Step 2 — Export PyRIT memory to JSON

```python
import json
from pyrit.memory import CentralMemory          # accessor names vary by PyRIT version
memory = CentralMemory.get_memory_instance()
pieces = memory.get_prompt_request_pieces()      # adjust to your version if needed
json.dump([p.to_dict() for p in pieces], open("pyrit_pieces.json", "w"), ensure_ascii=False)
```

(Or point the adapter at a DuckDB memory file directly with `--from-duckdb`, if `pyrit` is
installed. The JSON path is more robust across PyRIT versions.)

## One-command path (recommended)

The harness can consume a PyRIT memory export **directly** — no intermediate JSONL file:

```
python process_utils/robustness_eval.py \
    --targets targets.example.json \
    --pyrit-json pyrit_pieces.json \
    --pyrit-objective "fallback objective if not in PyRIT labels" \
    --pyrit-image-base /path/to/pyrit/image/store \
    --benign-dir exports/benign_multimodal \
    --judge-model openai/gpt-4o-mini \
    --output reports/robustness_report
```

Use `--pyrit-duckdb path.db` instead of `--pyrit-json` to read a DuckDB memory file (requires
`pyrit` installed). Steps 3–4 below are the explicit two-stage version if you prefer to inspect
the converted transcripts first.

## Step 3 — Convert to the harness transcript schema

```
python tools/pyrit_export_adapter.py \
    --from-json pyrit_pieces.json \
    --output exports/pyrit_transcripts.jsonl \
    --objective "OPTIONAL fallback objective if not in PyRIT labels" \
    --image-base /path/to/pyrit/image/store
```

The adapter groups pieces by conversation, extracts the **attacker (user) turns** (merging a
text + image turn into one), tags each conversation with its `attack_family`
(`pyrit_crescendo` / `pyrit_pair` / `pyrit_tap` / `pyrit_redteam`), carries the objective, and
preserves PyRIT's captured responses + scores in `attacker_metadata` for the audit trail.

If your PyRIT export uses different field names, override them with `--field-role`,
`--field-value`, `--field-datatype`, `--field-conversation`, `--field-sequence`,
`--field-orchestrator`, `--field-labels`.

## Step 4 — Score against your targets and build the report

```
python process_utils/robustness_eval.py \
    --targets targets.example.json \
    --transcripts exports/pyrit_transcripts.jsonl \
    --attack-dir results/comet_run:comet_entanglement \
    --benign-dir exports/benign_multimodal \
    --judge-model openai/gpt-4o-mini \
    --output reports/robustness_report
```

The report breaks ASR down **per attack family** (so you see which techniques your defense holds
against), shows the **undefended-vs-defended delta**, and the **benign over-refusal** cost. All
numbers use the strict, indeterminate-aware scorer (see `AUDIT_NOTES.md`).

## Replay semantics (important for interpreting results)

The harness **replays the captured attacker turns** against each target config. It does not
re-run PyRIT's adaptive loop. So a transcript is a *frozen* attack script:

- Replaying it against the **undefended** target should roughly reproduce the original breach.
- Replaying the same script against the **defended** target shows whether the guardrail blocks
  that specific attack path — which is exactly the robustness question.

A fully rigorous "does the defense survive an attacker that re-adapts *to the defense*" test
means running PyRIT again with the defended model as its target, then scoring that transcript.
Do both and report them separately; the literature (`../literature/02-defenses.md`) treats
adaptive-to-the-defense evaluation as the stronger claim.

## Before citing any number

- Log the judge model/temperature (the harness does this in `evaluation_info`).
- Report `success_rate` **with** `indeterminate_rate` and the strict lower bound.
- Add a human-agreement spot check on a sample of the judge's verdicts.
