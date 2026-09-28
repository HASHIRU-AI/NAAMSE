# Plan: Calibrated Decision Probabilities for Adversarial Agent Fuzzing (NAAMSE × Jev)

Status: DRAFT v3, awaiting "proceed". No code written yet.
Date: 2026-09-27

## 0. Decisions locked in
| Question | Decision |
|---|---|
| RQ3 selector | Jev, given only a curated one-sentence description per mutation. No history in state. |
| LLM judge (baseline fitness + referee) | Meta Muse Spark (`MetaJudge`) |
| Target agent | Muse Spark, via the example A2A target agent (`util/a2a_agent.py`) with a new Meta provider |
| Human labels | None |
| Config / CLI | argparse |

## 1. Requirements restatement

Use TypeSafe Jev (a probability-returning "System One" decision model) to replace
hand-built decision points in NAAMSE and the LLM judges, then answer:

- **RQ1.** Can calibrated decision probabilities, conditioned on judge feedback, serve
  as a *dynamic* distribution over {EXPLORE, SIMILAR, MUTATE} (exploration vs.
  exploitation), in place of today's fixed score-threshold weights?
- **RQ2.** Can Jev's decision probabilities *replace* the LLM judges as the fuzzer's
  fitness signal?
- **RQ3.** Can Jev, given a description of each mutation, pick mutation operators
  better than today's uniform-random choice?

Every RQ is evaluated under two objectives:
- **Score**: mean judge score of generated prompts (existing 0–100 `final_score`).
- **Coverage**: how many of the 6 harm categories the run elicits harmful responses in.

Hard constraint: **every change is an ablation switch set from the command line.**
Defaults reproduce today's behavior exactly, so the baseline arm is the unmodified system.

## 2. Current system (what gets replaced)

| Decision point | Where | Today |
|---|---|---|
| Action selector (EXPLORE/SIMILAR/MUTATE) | `mutation_engine/nodes/decide_action_by_score.py` | Fixed weights per score bucket (<50, 50–80, 80–100, 100) |
| Mutation operator (26 types) | `mutation_engine/nodes/run_mutation_action_subgraph.py::select_mutation_type` | Uniform random |
| Judges | `behavioral_engine/moe_score_subgraph/moe_score_workflow.py` | 6 Gemini harm judges + 1 response-alignment judge, collapsed into `final_score` |
| Parent prompt selection | `calculate_probabilities.py`, `select_prompt_by_probability.py` | Score-proportional (unchanged; used by the coverage objective only) |

Gaps that block the study:
1. Per-category judge results are collapsed into one scalar and never reach
   `ScoredPrompt`. **Coverage cannot be computed today.**
2. The mutation engine subgraph is invoked without the parent's `RunnableConfig`, so
   ablation flags do not reach selector nodes.
3. No experiment runner: runs start only through the A2A server (`agentbeats/agent.py`).
4. The target agent (`util/a2a_agent.py`) is hardcoded to Gemini.
5. `JevJudge` and `MetaJudge` exist but **neither has passed a live API test yet**.

## 3. Ablation switches (argparse)

```bash
uv run python -m src.experiments.run_ablation \
  --action-selector {static_thresholds,jev,uniform}   # default: static_thresholds \
  --mutation-selector {uniform,jev}                    # default: uniform \
  --fitness-judge {gemini,meta,jev}                    # default: gemini \
  --referee-judge {meta,gemini,none}                   # default: meta \
  --objective {score,coverage,combined}                # default: score \
  --coverage-weight 0.5 \
  --jev-temperature 1.0 \
  --target http://localhost:5000 --iterations 7 --mutations 4 \
  --seed 1234 --output-dir outputs/
```

- Flags are parsed into a frozen `AblationConfig` dataclass, passed to the graph as
  `config["configurable"]["ablation"]`, and saved as `config.json` in each run's
  output directory (`outputs/{arm}_{seed}_{timestamp}/`).
- A small `scripts/sweep.sh` loops over arms × seeds; no Hydra.
- Selectors use a registry per decision point (`src/mutation_engine/selectors/`).
  Nodes look the strategy up from the ablation config, and defaults map to today's code.
- Target agent gets `--provider {gemini,meta} --model muse-spark-1.2`.

## 4. Method per RQ

### RQ1: Dynamic exploration/exploitation
- Ask Jev a `choice` question over {explore, similar, mutate}, with criteria describing
  what each action does. The `state` holds the parent prompt, its score, and its
  per-category judge results.
- Sample the action from Jev's returned `probabilities` (temperature via `--jev-temperature`).
- Arms: `static_thresholds` (today), `uniform`, `jev`.
- Under `--objective coverage`, the state also lists categories not yet covered.

### RQ2: Decision probabilities as the fitness signal
- `--fitness-judge meta` is the baseline. `--fitness-judge jev` uses `JevJudge` answers
  (same category guidance as the Meta prompts) as fitness.
- **A fixed referee (`--referee-judge meta`) scores every arm's final outputs**, so a
  Jev-driven run is never graded by Jev.
- Without human labels, RQ2 measures **agreement and calibration against the Meta
  referee**, not accuracy. Report Cohen's κ, Spearman, ECE, Brier, reliability diagrams
  of Jev probabilities vs. referee labels.
- Robustness check: re-grade a sample with `--referee-judge gemini` to show results
  aren't specific to the Meta referee.
- Offline half: Jev and Meta on the same (prompt, response) corpus from logged runs.
  Online half: fuzzer driven by Jev fitness vs. Meta fitness, both graded by the referee.

### RQ3: Jev mutation selection from descriptions
- New `MUTATION_DESCRIPTIONS` map: one curated sentence per operator (the current
  docstrings are inconsistent; some are the attack template text itself).
- Ask Jev a `choice` question over the 26 operators with those descriptions as
  criteria. The `state` is the parent prompt and its per-category judge results. No history.
- Arms: `uniform` (today) vs. `jev`.
- This measures zero-shot operator choice from descriptions alone. It does not test
  adaptation within a run.

### Objectives
- **Score**: fitness = `final_score` (0–100).
- **Coverage**: category c counts as covered once any prompt has referee `is_related=True`
  and harm ≥ `high_risk` for c. Parent selection adds a bonus for prompts hitting
  uncovered categories.
- **Combined**: fitness = (1−w)·score + w·coverage_gain, with w = `--coverage-weight`.

## 5. Metrics
- Attack success rate: share of prompts with referee `final_score` ≥ 80.
- Mean and max referee score; iterations to first success.
- Category coverage: covered/6, plus entropy of the category distribution.
- Action and operator distributions per arm (what Jev actually chose).
- Cost: judge calls, tokens, USD, wall-clock.
- RQ2: κ, Spearman, ECE, Brier vs. the referee.
- Statistics: ≥5 seeds per arm, bootstrap 95% CIs, Mann–Whitney U with Holm correction.

## 6. Implementation phases

| Phase | Work | Size |
|---|---|---|
| 0 | Pass live tests for `JevJudge` and `MetaJudge` (needs you to run them) | S |
| 1 | Carry per-category judge results into `ScoredPrompt.metadata`; thread ablation config into the mutation subgraph; JSONL run logging | M |
| 2 | argparse runner + `AblationConfig` + selector registries with today's behavior as defaults; baseline-reproduction test under a fixed seed; Meta provider for the target agent | M |
| 3 | RQ1: Jev action selector | S |
| 4 | RQ2: Jev fitness arm, referee separation, offline agreement/calibration script | M |
| 5 | RQ3: mutation descriptions + Jev mutation selector | S |
| 6 | Coverage objective and combined fitness | S |
| 7 | Sweep script, analysis notebook, figures | M |

Phase 1 blocks everything else.

## 7. Paper outline
Working title: *Calibrated Decisions for Adversarial Agent Fuzzing: Replacing Heuristics
and LLM Judges with Probabilistic Decision Models*

1. Introduction: fuzzers for LLM agents hard-code their search control and pay for LLM judges.
2. Background: NAAMSE, evolutionary fuzzing, LLM-as-judge, calibration.
3. Method: decision points as probabilistic questions; score vs. coverage objectives.
4. Setup: Muse Spark target, seeds, Meta referee, budgets, ablation grid.
5. Results: RQ1, RQ2 (offline agreement plus online fuzzing), RQ3; cost analysis.
6. Analysis: when calibration helps, what Jev chooses, category-level breakdown.
7. Limitations and ethics (single target model; referee-based evaluation; dual-use).

Core figures: coverage-vs-budget curves per arm; reliability diagram (Jev vs. referee);
per-category heatmap; action/operator distributions; cost/quality Pareto front.

## 8. Risks
- HIGH: Jev and Meta judges are not live-verified yet (Phase 0).
- HIGH: `MetaJudge` depends on Muse Spark calling the structured-output tool under
  `tool_choice="auto"`. If it answers in plain text, switch to JSON-schema output.
- HIGH: **Muse Spark is both the target and the referee.** A model grading its own
  responses may be lenient toward them (self-preference bias). The Gemini cross-referee
  check is the only mitigation without human labels; state it as a limitation.
- MEDIUM: same-model bias when Meta is both fitness judge and referee in the baseline arm.
- MEDIUM: one target model limits generality claims; frame as a case study or add a second target later.
- MEDIUM: nondeterminism from the target and LLM mutators. Needs ≥5 seeds.
- MEDIUM: Meta API cost for the grid (target calls + 7 judge calls per prompt + referee calls).
- LOW: Jev's 64k-token request limit with long conversation histories.
