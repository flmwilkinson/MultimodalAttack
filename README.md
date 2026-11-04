# COMET: Cross-Modal Entanglement Attack

![Overview](pic/overlook.png)

## Adoption
This is an adoption of the below algorithm that implements the full local pipeline. You need to install ollama
ollama version is 0.32.5
This version allows for a text to image model, e.g. x/z-image-turbo:latest which provides pretty decent quality
The target LVLM is qwen3-vl:30b-a3b-instruct
The other models can be located in .env configuration input file

## Abstract
Vision-Language Models (VLMs) with multimodal reasoning capabilities are high-value attack targets, given their potential for handling complex multimodal harmful tasks. Mainstream black-box jailbreak attacks on VLMs work by distributing malicious clues across modalities to disperse model attention and bypass safety alignment mechanisms. However, these adversarial attacks rely on simple and fixed image-text combinations that lack scalable attack complexity, limiting their effectiveness for red-teaming VLMs’ continuously evolving reasoning capabilities.

We propose COMET (Cross-Modal Entanglement Attack), a scalable approach that extends and entangles information clues across modalities to exceed VLMs’ trained and generalized safety alignment patterns for jailbreak. Specifically, knowledge-scalable reframing extends harmful tasks into multi-hop chain instructions; cross-modal clue entangling migrates visualizable entities into images to build multimodal reasoning links; and cross-modal scenario nesting uses multimodal contextual instructions to steer VLMs toward detailed harmful outputs. Experiments across multiple advanced VLMs show COMET achieves over 94% attack success rate, outperforming the best baseline by 29%.

Disclaimer: This study contains AI-generated content that may be offensive.

> **Evaluation-integrity note (added 2026-11-01).** The "over 94% ASR" figure above is the
> original authors' *claim* and is **not** a number produced by this repo's default scorer as
> shipped. The StrongREJECT scorer here previously counted several non-measurements (judge
> refusing to grade, unparseable judge output, no judge reply, scoring exceptions) as
> successes, biasing ASR upward. That has been corrected — see **`AUDIT_NOTES.md`**. Any ASR
> you cite from this tool must be reported with its judge model/temperature, retry policy
> (best-of-N vs single-shot), success definition, and the `indeterminate_rate` of the run.
> Background: `../literature/03-evaluation-and-judges.md`.

## Features
- Scalable cross-modal entanglement attack pipeline for VLMs
- Modular components: entity extraction, text redaction/rewriting, payload assembly, VLM recognition
- StrongREJECT evaluation tools and small datasets for reproducible scoring
- Clean configuration: no hardcoded secrets; environment-driven APIs

## Repository Structure
- `core/`: core modules (entity extraction, text rewriting, payload assembly, VLM recognition)
- `transmuter/`: rubric generation, services, configs (`setting/llm_config.py`: DeepSeek default, OpenAI GPT-4o fallback)
- `process_utils/`: StrongREJECT evaluator, payload evaluation, schemas, and `StrongReject/` datasets
- `tools/`: image generation and rubric utilities
- `results/`: curated examples and JSON subsets
  - `anthropic/claude-haiku-4.5/attack payload {1..5}/`: five full successful cases
  - `strongreject_top10_20251112.json`, `strongreject_top5_20251113.json`: top subsets (IDs removed)
- Top-level: `main.py`, `eval.py`, `.env.example`, `requirements.txt`

## Quick Start (Offline)
- Evaluate curated examples without any API keys:
- `python eval.py --dataset COMET-Review-Package\results\anthropic\claude-haiku-4.5`

## Online Model Configuration (Optional)
- Set environment variables if using online APIs:
- DeepSeek: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com/v1`)
- OpenAI: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- Default selection: primary `deepseek-chat`, fallback `openai/gpt-4o`

## Installation
- Minimal dependencies: see `requirements.txt`
- For online use or image generation, install respective SDKs as needed

## Howto run
python main.py --mode batch --input data/SafeBench-One.txt --output logs/attack_run3/ --with-mmjailbreak -v    

## Some results are stored in
logs/attack_run4/query_1_20261027_230143

## Robustness evaluation (for a defensibility / CRO report)

`process_utils/robustness_eval.py` is a **measurement** harness that answers "is the deployment
robust to the known multimodal attack taxonomy, and how much does our guardrail buy us?" It runs
a fixed attack battery you supply against one or more targets (owned, with the defense off and
on; and/or a third-party API), scores every response with the strict StrongREJECT instrument,
and writes a CRO-ready report (`.json` + `.md`) containing per-family success rates, the
**undefended-vs-defended delta**, a **benign over-refusal** measurement, and an OWASP/NIST
mapping stub.

```
python process_utils/robustness_eval.py \
    --targets targets.example.json \
    --attack-dir results/comet_run:comet_entanglement \
    --attack-dir exports/mm_safetybench:mm_safetybench \
    --transcripts exports/pyrit_crescendo.jsonl \
    --benign-dir exports/benign_multimodal \
    --judge-model openai/gpt-4o-mini \
    --output reports/robustness_report
```

**Testing adaptive / multi-turn (Crescendo-style) attacks.** This harness does not generate or
escalate attacks — it replays and scores fixed inputs. To exercise adaptive adversaries, run
Microsoft **PyRIT** (multimodal orchestrators + Crescendo; see
`../literature/06-tooling-and-governance.md`) under your own authorization, export the resulting
attacker-turn transcripts as JSONL (schema: `transcripts.schema.example.jsonl`), and pass them
via `--transcripts`. The adaptiveness lives in PyRIT; the harness measures the outcome. This
keeps the methodology recognised, reproducible, and citable.

See `AUDIT_NOTES.md` for the evaluation-integrity fixes the scoring depends on, and
`PYRIT_INTEGRATION.md` for the end-to-end PyRIT → adapter → harness → report workflow
(`tools/pyrit_export_adapter.py` converts PyRIT orchestrator output into the harness schema).

## Ethics & Safety
- No hardcoded secrets; all API access via environment variables
- All examples are for research and review; no personal data included

## Citation
Please cite COMET if you find it useful in your research. A formal bib entry will be provided upon paper release.

## License
This review package is provided for academic review purposes. Please contact the authors for licensing details.
