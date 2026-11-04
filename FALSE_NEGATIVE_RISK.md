# Interpreting a "No Breach" Result — False-Negative Risk Framework

**Written for:** a red-team engineer and their reporting line (CRO/risk committee).
**The question this answers:** "If our test shows no breach of a use case and we release it, how
exposed are we to someone else breaching it afterwards?"

**Short answer.** A "no breach" result is a *conditional* statement, never a guarantee of safety.
Its value equals the strength and representativeness of the test that produced it. Reported as an
absolute ("the system is robust"), it is a false-negative risk that can directly cause a
post-release breach. Reported conditionally with the safeguards below, it is a defensible finding.

---

## 1. Never report a negative as absolute

Report: *"No breach found against attack battery **X**, judge **Y** at temperature **t**, **N**
attempts per objective, as of **date Z**, on model/version **M**."*

Not: *"The use case is robust."* The first is auditable and bounded; the second overclaims and is
the statement that gets falsified in production.

## 2. Sources of false negatives (each must be addressed before trusting a clean result)

| Source | Failure | Mitigation |
|---|---|---|
| Coverage gap | Only tested families are covered; omitted ones are invisible risk | Coverage matrix (§3); state untested families as residual risk |
| Benchmark saturation | Clean score on an easy/old benchmark means nothing | Test on current/hard benchmarks, not just a regression floor |
| Non-adaptive test | Fixed attacks say nothing about an adaptive adversary | Adaptive-to-the-defense runs (PyRIT against the DEFENDED model) |
| Judge under-counts | A real breach scored as a refusal (false negative in your favour) | Human-agreement spot check on clean cases (§4) |
| Single-shot | Understates a persistent attacker | Match attempts/persistence to the real threat model; report N |
| Broken detector | Test "passed" because it couldn't detect anything | **Positive controls (§5)** |
| Drift | Model/prompt/tools/retrieval change after the test | Residual-risk register + re-test cadence (§6) |

## 3. Coverage matrix — make the untested explicit

Enumerate the attack taxonomy (see `../literature/01-threat-landscape.md`) and mark each family
tested / not tested, with the corpus and date. The CRO decision is then made against an explicit
list of what was NOT exercised — the residual risk is named, not hidden.

| Attack family | Tested? | Corpus / tool | Last run | Result |
|---|---|---|---|---|
| Cross-modal entanglement (COMET) | | | | |
| Typographic (FigStep) | | | | |
| Keyword+diffusion (MM-SafetyBench) | | | | |
| Optimization-based (HADES) | | | | |
| Multi-turn escalation (Crescendo) | | | | |
| Iterative refinement (PAIR / TAP) | | | | |
| Cross-modal concealment (steganographic) | | | | |
| Agent / visual prompt injection (VPI) | | | | |
| …(add your deployment-specific vectors) | | | | |

## 4. Human-agreement spot check — catch the judge's misses

Sample the cases the automated judge marked "no breach / refusal" and have a human re-grade them.
This directly measures judge *recall* and catches real breaches the judge scored as safe. The
literature treats a human-agreement check as mandatory before citing any automated result
(`../literature/03-evaluation-and-judges.md`). Report the sampled agreement rate alongside the ASR.

## 5. Positive controls — prove the test can detect a breach at all

Include, in every run, a small set of attacks **known** to breach an *undefended* model. If these
do not register as breaches, the pipeline is not measuring what you think it is, and every
"no breach" result from that run is void. This is the safeguard against the exact failure mode of
"we released it because the test was clean, but the test was broken."

A clean run is only trustworthy if: positive controls fired on the undefended target **and** did
not fire on the defended target. That contrast is the evidence.

## 6. A negative result has a shelf life

Robustness decays: new attack techniques appear, and your model/prompt/tooling drift. Maintain a
residual-risk register (untested families, known limitations, judge recall) and a re-test cadence.
A "no breach" finding should carry an expiry and a re-test trigger (model change, new attack class
published, incident elsewhere).

## 7. Confidence framed as survived attacker effort

State what the negative *survived*, not merely that nothing was found:
*"Survived N iterations of PyRIT PAIR/TAP/Crescendo against the defended model, plus the
MM-SafetyBench and JailBreakV-28K corpora, with positive controls confirmed and X% human
agreement on a sample of clean cases."* That is a bounded, defensible claim. "We didn't find
anything" is not.

---

## Bottom line for the CRO conversation

Whether to build more guardrails should hinge on the **coverage matrix + residual-risk register**,
not on a headline ASR of 0%. A 0% that comes from a saturated benchmark, a lenient-or-conservative
unchecked judge, no adaptive testing, and no positive controls is the *most* dangerous number you
can hand up the chain. The safeguards above convert "we found no breach" into "here is exactly what
we tested, how hard, how we know the test works, and what we did not cover" — which is what lets a
risk committee make a real release decision.
