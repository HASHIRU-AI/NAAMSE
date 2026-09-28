# Probabilities Without Payoff: Replacing Hand-Built Decisions in Evolutionary Agent Fuzzing with a Calibrated Decision Model

*Draft mini-paper, revised 2026-09-28. All numbers are computed from the 40 runs in `outputs/` by
the scripts listed in Appendix A. Figures are in `figures/`; tables without a figure are the CSV and
JSON files named in each section.*

## Abstract

Evolutionary fuzzers for LLM agents (NAAMSE among them) steer their search with hand-built
control: fixed score thresholds choose between exploring, exploiting, and mutating; mutation
operators are drawn uniformly at random; and a panel of LLM judges supplies the fitness signal.
Whether a probability-returning decision model can replace these components under a fixed attack
budget has not been tested. We replace each decision point in NAAMSE with TypeSafe Jev (a model
that returns a probability for every answer option) and evaluate eight ablation arms (40 runs,
1,120 attack prompts) against a Muse Spark target, backed by five budget-free or budget-light
follow-ups: off-policy re-estimation, an operator-description ablation (350 Jev calls, no target
queries), a common referee for the Jev-judged arm, a blind cross-model annotation of 103 prompts,
and an audit of the fuzzer's own randomness and mutator.
No arm outperforms the baseline (every Holm-adjusted p ≥ 0.50). Three estimators bound Jev's
action policy to within two score points of the static thresholds, and an oracle policy gains
under three. Jev's zero-shot operator preferences follow the wording of the operator descriptions
(ρ = 0.83 when descriptions are swapped between operators) and are uncorrelated with operator
outcomes (ρ = −0.11). Under a common referee, Jev as fitness judge is indistinguishable from Meta
(+3.3 [−3.8, +11.1] score points) and agrees with it substantially (quadratic κ = 0.76), while
calling 13 points fewer refusals. A blind cross-model annotation of 103 prompts confirms the
judges' alignment labels (κ = 0.74 against Meta) but none of their high-risk or harmful verdicts:
9 of the 13 successes are full compliance with low-risk content, and the annotator rates the other
4 as low risk or not harmful. The audit surfaces two properties of the fuzzer that shape every
arm: 27% of mutated prompts are the mutation LLM's own refusal, and a shared per-task seed couples
the action draw to the operator and corpus draws. Therefore, under a small budget against a
refusal-heavy target, decision calibration is not the bottleneck: the attack material (operator,
attack family, and whether the mutator complies) carries the leverage. A refusal-gated referee
cuts judge calls by 53% without changing any score. Analysis code and derived tables are released
with NAAMSE.

## 1 Introduction

Automated red-teaming of LLMs and agents has moved from static benchmarks toward adaptive search.
Mutation-based fuzzers evolve jailbreak templates (GPTFuzzer [2]), attacker LLMs refine prompts
iteratively (PAIR [3]), and NAAMSE [1] reframes agent security evaluation as feedback-driven
optimization over a corpus of seed attacks. Each of these systems hard-codes the control of its
search. In NAAMSE, three components carry that control: (i) a *score-threshold policy* that
chooses among EXPLORE (sample a new corpus prompt), SIMILAR (sample a neighbor of the parent), and
MUTATE (apply an operator to the parent); (ii) a *uniform draw* over 26 mutation operators; and
(iii) a panel of LLM judges (LLM-as-a-judge [4]) whose verdicts form the fitness signal.

These components are heuristics, and heuristics invite replacement. We hypothesize that a decision
model returning calibrated probabilities (in the sense of Guo et al. [6]) improves the search along
three axes, viz. action selection (RQ1), fitness judging (RQ2), and operator selection (RQ3): it
conditions each choice on the parent prompt and its judge verdicts, exposes its uncertainty, and
scores responses without free-text generation.

A first sweep answered this hypothesis in the negative, but left three questions a reviewer would
rightly press on. The Jev-judged arm was graded by Jev itself, so RQ2 had no common scale. The
pooled action estimates mixed decisions from different selectors, so they could be confounded.
And the operator analysis rested on 35 Jev picks whose distribution did not match Jev's own logged
probabilities. To address these limitations, we add five follow-ups that require no new
target queries and at most 458 judge calls in total (Table 2), and we trace the probability mismatch to the fuzzer's seeding. In this paper, we
make the following contributions:

- **A controlled ablation of fuzzer control:** eight arms isolating Jev as action selector,
  operator selector, and fitness judge, each under a score and a coverage objective (40 runs,
  1,120 attack prompts, identical budgets).
- **A bounded null for learned action selection, with a hidden cost:** direct-method, self-normalized
  importance sampling (SNIPS), and doubly robust (DR) estimates all bound Jev's policy to within two
  score points of the static thresholds, and corpus-cluster analysis shows that the policy narrows
  attack diversity by roughly a quarter.
- **What drives zero-shot operator selection:** a five-condition description ablation shows that
  Jev's operator preferences are set by the operator descriptions rather than the operator names or
  the parent, favor operators that perform poorly, and, when given outcome statistics, chase
  estimates from one or two samples.
- **A common-referee evaluation of Jev as a judge at half the cost:** a refusal-gated Meta referee
  places the Jev-judged arm on the common scale using 458 instead of 980 judge calls, and yields
  agreement and calibration estimates for Jev's verdicts; a blind cross-model annotation of 103
  prompts then checks both judges and the severity of the study's successes.
- **Two fuzzer properties that shape every arm:** a mutator-refusal rate of 27% among mutated
  prompts, and a shared per-task seed that couples the action draw to the operator and corpus
  draws. We quantify the effect of each on the reported results.

## 2 Background

**NAAMSE.** Each iteration selects parent prompts in proportion to fitness, applies one action per
parent, sends the children to the target over A2A, and scores the responses. The final score
combines a response-alignment term (25, 42, 58, or 75 for full refusal, weak refusal, weak
acceptance, and full acceptance) with the maximum harm term across six categories (−20, 5, 15, or
30 for not harmful through harmful), boosted by at most 10% for PII. The score is therefore
discrete in practice: refusals land near 20, partial compliance near 37, benign compliance near
54, and any score of 80 or above requires a judge to find harm. We call a prompt with score ≥ 80 a
*success*. The threshold is low: a full acceptance with one low_risk category scores exactly 80, so
a success need not contain high-risk content (Section 5.5).

**Static action policy.** For parent scores below 50, the weights over (EXPLORE, SIMILAR, MUTATE)
are (0.7, 0.2, 0.1); for 50 to 80, (0.1, 0.7, 0.2); for 80 to 100, (0.1, 0.2, 0.7).

**Per-task seeding.** NAAMSE draws one seed per parallel task, and each decision in that task
instantiates a fresh generator from it (`random.Random(task_seed)`). The action draw, the operator
draw, and the EXPLORE corpus draw therefore all begin from the same first variate u. Two
consequences follow, and both hold in every arm of this study. First, MUTATE (the last action in
the cumulative distribution) is taken only for large u, and the operator is then drawn with the
same u over options sorted by name, so realized operators concentrate at the end of the alphabet
whenever the selector's distribution is non-uniform. Second, EXPLORE (the first action) is taken
only for u < P(explore), and the corpus offset is approximately u × 2^17, so EXPLORE samples only
the first P(explore) fraction of the 129,454-prompt corpus in storage order. We treat this as the
configured behavior of the system under test and measure its effect (Sections 5.2 and 5.3).

**Jev.** Jev answers typed questions (choice, yes/no) with a probability per option. We pose each
decision point as a choice question: the `state` carries the parent prompt, its score, and its
per-category judge verdicts, and the action is sampled from the returned probabilities
(temperature 1.0). For operator selection, the options are the 26 operators, each described by one
curated sentence, with no run history. As a judge, Jev answers the same category guidance as the
Meta judges, and a category counts as related when P(related) ≥ 0.5.

## 3 Threat Model and Setup

**Adversary.** A black-box attacker with query access to a deployed agent over A2A, no access to
weights or system prompt, and a fixed budget of 7 iterations × 4 mutations (28 attack prompts per
run). The attacker's goal is to elicit responses judged harmful in any of six categories
(disinformation, illegal goods and services, hate and harassment, non-violent crime, violence,
sexually explicit content).

**Target and judges.** The target is Muse Spark behind the NAAMSE example A2A agent. The mutation
LLM and the default fitness judges are Meta models. The Fit=Jev arm uses Jev judges instead.

**Arms.** Table 1 lists the eight arms. Each ran with seeds 1 to 5.

| Arm | Action selector | Operator selector | Fitness judge | Objective |
|---|---|---|---|---|
| Baseline | static thresholds | uniform | Meta | score |
| Act=uniform | uniform | uniform | Meta | score |
| Act=Jev | **Jev** | uniform | Meta | score |
| Mut=Jev | static thresholds | **Jev** | Meta | score |
| Fit=Jev | static thresholds | uniform | **Jev** | score |
| Cov baseline | static thresholds | uniform | Meta | coverage |
| Cov + Act=Jev | **Jev** | uniform | Meta | coverage |
| Cov + Mut=Jev | static thresholds | **Jev** | Meta | coverage |

*Table 1: Ablation arms. Bold marks the component replaced by Jev.*

**Follow-ups.** None of the follow-ups queries the target again.
(i) *Off-policy re-estimation* uses the logged behavior probabilities of every action decision.
(ii) *Description ablation* replays the 66 parents that were mutated in the score-objective,
Meta-judged arms to Jev's operator question under five framings of the options (Section 5.2).
(iii) *Common referee* re-scores the 140 Fit=Jev prompts with the Meta judges. The referee is
refusal-gated: it runs the alignment judge first and calls the six harm judges only when the
response is not a refusal. In the sweep, all 754 responses judged as refusals received "not
harmful" in every category, so gating reproduces every stored score exactly.
(iv) *Randomness and mutator audit* inspects the mutated prompts and the corpus offsets of EXPLORE
draws.
(v) *Cross-model annotation* labels a stratified sample of 103 prompts (every own-judge success,
45 full acceptances scoring 50 to 80, 15 weak acceptances, and 30 refusals) with an annotator from
a model family outside both judges (Claude). The sheet hides every judge verdict and shuffles the
strata; four annotator instances label disjoint quarters of it with the judges' own rubrics, and
the key is joined only after all labels are written.

| Follow-up | Target queries | Judge calls | Jev calls | Other |
|---|---|---|---|---|
| (i) Off-policy re-estimation | 0 | 0 | 0 | logged probabilities |
| (ii) Description ablation | 0 | 0 | 350 (+8 retries) | 66 replayed parents |
| (iii) Common referee (gated) | 0 | 458 (980 ungated) | 0 | 140 prompts |
| (iv) Randomness and mutator audit | 0 | 0 | 0 | corpus database |
| (v) Cross-model annotation | 0 | 0 | 0 | 103 blind labels |
| *Original sweep, for reference* | *1,120* | *about 7,840* | *decision calls* | *40 runs* |

*Table 2: Cost of the follow-ups. The original sweep used seven judge calls per prompt.*

**Metrics.** Mean and maximum judge score per run, number of successes, harm-category coverage
(categories with a related response judged high_risk or harmful), refusal rate, and action shares.

**Statistics.** The unit of analysis is the run. Same-seed runs do not share starting prompts
across arms (each selector consumes the random stream differently), so all arm contrasts are
unpaired: exact two-sided permutation tests on the difference of run means (252 splits), 95%
bootstrap intervals, Cliff's δ, and Holm correction across seven planned contrasts per metric. For
pooled analyses, intervals come from a cluster bootstrap that resamples whole runs.

## 4 Arm-Level Results

**No arm is distinguishable from the baseline.** Figure 1 shows per-run outcomes. The largest
mean-score contrast among Meta-judged arms is Act=Jev − Baseline = +2.7 [−4.2, +9.3] (p = 0.50,
δ = 0.28). Across mean score, maximum score, successes, coverage, and refusal rate, every
Holm-adjusted p-value is 1.00, except maximum score for Cov + Mut=Jev versus its baseline (0.50,
driven by a single run reaching 100). The Fit=Jev arm is compared on the common referee scale in
Section 5.5.

**Successes are rare and clustered.** Under each arm's own judge, only 13 of 1,120 prompts (1.2%)
reach score ≥ 80. They come from 6 of 40 runs, and three runs (Baseline seed 2, Act=Jev seed 4,
Fit=Jev seed 2) contain 10 of them. Coverage is zero in 35 of 40 runs. The target refuses about 70%
of attack prompts (Baseline refusal rate 0.69 ± 0.15), and the modal non-refused outcome is benign
compliance (score ≈ 54; Figure 6). The successes are also mild: 9 of the 13 are full acceptances
whose strongest verdict is low_risk (for example, suggestive pickup lines and non-graphic fictional
scenarios), and an independent annotator finds no high-risk content in any of them (Section 5.5).

**The search stalls rather than running out of budget.** Best-so-far curves flatten by iteration 2
to 3 in every arm (Figure 5). Act=Jev starts higher at iteration 0 only because two initial EXPLORE
samples scored 80, before any Jev decision could matter.

Taken together, five runs per arm cannot resolve differences smaller than several score points at
this success rate. The remaining sections recover power by pooling individual decisions and by
asking offline questions that need no new target queries.

## 5 Where the Leverage Is (and Is Not)

### 5.1 RQ1: Jev learns a sensible policy that does not pay off

**Jev's policy is monotone in parent score.** Figure 3 plots Jev's mean action probabilities by
parent-score bucket against the static weights. P(MUTATE) rises from 0.14 (parents below 30) to
0.80 (parents at 80 or above), and P(EXPLORE) falls from 0.37 to 0.04. The direction matches the
static policy, but Jev prefers SIMILAR where the static policy prefers EXPLORE (parents below 50).
As a result, Act=Jev shifts its budget from EXPLORE (0.56 → 0.31 of prompts) toward SIMILAR
(0.29 → 0.44) and MUTATE (0.16 → 0.26).

**Pooled transitions show where Jev is right and where it is wrong.** We pool 480 transitions
(iteration ≥ 1) from the four score-objective, Meta-judged arms and measure the child produced by
each action per parent bucket (Figure 7, Table 3).

| Parent score | EXPLORE | SIMILAR | MUTATE | Static prefers | Jev prefers |
|---|---|---|---|---|---|
| < 30 (n = 257) | 25.0 (0.79) | **28.9** (0.66) | 22.4 (0.87) | EXPLORE | **SIMILAR** |
| 30 to 50 (n = 73) | **27.5** (0.79) | 21.7 (0.92) | 28.6 (0.50), n = 6 | **EXPLORE** | SIMILAR |
| 50 to 80 (n = 130) | 23.9 (0.86) | **41.1** (0.35) | 37.5 (0.44) | **SIMILAR** | MUTATE ≈ SIMILAR |

*Table 3: Mean child score (refusal rate) by parent bucket and action. The ≥ 80 bucket (n = 20) is
omitted as too small.*

In the largest bucket, Jev's preference is correct and the static policy's is not. In the 30 to 50
bucket the reverse holds. The two errors cancel. Restricting the table to the Act=uniform arm,
where actions are randomized independently of the parent, reproduces the same ordering (SIMILAR
30.8, EXPLORE 26.2, MUTATE 23.6 for parents below 30; `uniform_arm_transitions.csv`), so the
pattern is not an artifact of pooling selectors.

**The policy effect is bounded under three estimators.** Every decision's behavior probability is
known (static weights, 1/3 for uniform, and Jev's logged probabilities), so we estimate each
policy's value with the direct method (DM), SNIPS, and DR [7] (Table 4, `offpolicy_values.csv`).

| Estimator | Static | Jev | Jev − static (score) | Jev − static (refusal rate) |
|---|---|---|---|---|
| DM | 29.7 | 29.8 | +0.1 [−0.7, +1.0] | −0.013 [−0.051, +0.025] |
| SNIPS | 29.1 | 30.0 | +0.9 [−0.2, +1.9] | −0.044 [−0.090, +0.006] |
| DR | 29.6 | 30.1 | +0.5 [−0.5, +1.5] | −0.032 [−0.077, +0.016] |

*Table 4: Offline policy values over 480 transitions (maximum importance weight 11.7). Intervals
resample runs.*

All three intervals exclude a gain of two points, roughly five times narrower than the arm-level
contrast. An in-sample oracle that always selects the best action per bucket reaches 32.6
[30.0, 35.7] versus 29.7 for the static policy: action selection can buy at most about three score
points here, although it can cut refusals from 0.67 to 0.54. One caveat applies to all four
values. Under the configured seeding, what EXPLORE returns depends on the policy's own P(explore)
(Section 5.3), and EXPLORE children drawn from earlier corpus regions score higher (Spearman
ρ = −0.29 between corpus offset and score, p < 10⁻⁵). The estimators assume a policy-independent
outcome for each action, so they are approximations for this system. Therefore, calibrated action
selection does not matter much in this setting because action selection itself does not matter
much.

### 5.2 RQ3: Zero-shot operator selection follows the descriptions, not the evidence

**Realized picks are not Jev's picks.** In the sweep, Jev chose MUTATE operators 35 times, and 33
of the realized picks were `semantic_steganography` (19) or `synonym` (14). Jev's own logged
probabilities do not support this: the realized operator carried a mean probability of 0.15, and
`synonym` carried 0.03 on average. The per-task seeding explains the gap (Section 2). Replaying
the logged probabilities through the configured seeding reproduces the concentration
(`semantic_steganography` 62%, `synonym` 22% in simulation), while sampling them independently
yields a broad preference (Table 5, Figure 13, `seed_coupling_operators.csv`). The realized picks are
therefore a property of NAAMSE's seeding applied to a non-uniform selector, not a preference of
Jev. The uniform selector is affected only mildly (in simulation, `synonym` rises from about 4% to
10% for parents scoring 50 to 80).

| Operator | Realized picks | Intended share (Jev probabilities) | Observed mean score (refusal) |
|---|---|---|---|
| `adversarial_prefix` | 0 | 0.26 | 21.4 (1.00), n = 6 |
| `persona_roleplay` | 1 | 0.19 | 25.6 (0.69), n = 16 |
| `semantic_steganography` | 19 | 0.19 | 23.3 (0.90), n = 30 |
| `contextual_framing` | 0 | 0.11 | 26.4 (0.67), n = 6 |
| `synonym` | 14 | 0.03 | 34.9 (0.50), n = 22 |

*Table 5: Jev operator selection, realized versus intended, over 35 MUTATE decisions. Observed
outcomes pool all 164 MUTATE prompts from the seven Meta-judged arms.*

**What Jev actually prefers is weak.** Jev's intended top three (`adversarial_prefix`,
`persona_roleplay`, `semantic_steganography`, 64% of the mass) rank among the weakest observed
operators, and `adversarial_prefix` was refused in all six of its uses. Operators Jev rarely
considers perform better: `artprompt` reaches 41.5 (n = 6, one success) and
`many_shot_jailbreaking` 58.5 (n = 2, one success). Operator means span more than 20 points, an
order of magnitude more than the action-selection headroom in Section 5.1.

**Descriptions set the preference.** To isolate what drives the choice, we replay the 66 parents to
Jev under five framings of the 26 options (Table 6, Figure 14, `jev_description_ablation.json`): the deployed
descriptions (*original*), names without descriptions (*names only*), shuffled anonymous labels
with the original descriptions (*anonymized*), names with descriptions permuted between operators
(*swapped*), and the original descriptions plus each operator's observed outcomes (*evidence*).

| Condition | Top operators (mean probability) | Effective operators | ρ with observed score |
|---|---|---|---|
| original | persona_roleplay 0.24, adversarial_prefix 0.22, semantic_steganography 0.16 | 9.3 | −0.11 |
| names only | deep_inception 0.19, persona_roleplay 0.18, contextual_framing 0.12 | 12.3 | −0.28 |
| anonymized | adversarial_prefix 0.22, persona_roleplay 0.22, semantic_steganography 0.16 | 9.7 | −0.08 |
| swapped | (mass follows the relocated descriptions) | 10.0 | +0.07 |
| evidence | many_shot_jailbreaking 0.60, payload_splitting 0.19, mathematical_attack 0.09 | 4.0 | +0.48 |

*Table 6: Jev operator preferences over 66 replayed parents. Effective operators is the perplexity
of the mean distribution. Repeated queries differ by a median of 0.02 in any probability.*

Three results follow. First, the description carries the decision: removing the names changes
almost nothing (ρ = 0.91 with the original), and when descriptions are permuted, Jev's
probabilities track where each description moved (ρ = 0.83) rather than the operator names
(ρ = 0.19). The key that inherits the `semantic_steganography` description receives 0.23 of the
mass, while the `semantic_steganography` key itself receives 0.02. Second, the parent matters
little: per-parent distributions stay close to the condition mean (mean Jensen-Shannon divergence
0.13 bits), so the choice is largely a fixed preference induced by the wording. Third, outcome
statistics do move Jev (ρ with observed score rises to +0.48), but toward the two operators with
the fewest observations (`many_shot_jailbreaking`, n = 2; `payload_splitting`, n = 1). Jev reads
the means and ignores the sample sizes, which is the behavior a bandit with an uncertainty bonus
exists to prevent.

**The mutator refuses more than a quarter of the time.** The mutation LLM is itself a safety-tuned
model, and it sometimes answers the request to transform an attack with a refusal. That refusal
text then becomes the "attack" sent to the target. A pattern audit of the 164 MUTATE prompts in
the Meta-judged arms flags 44 (27%) as mutator refusals, and some later operators wrap a refusal
produced earlier in the lineage (for example, a masked-token task whose template is a refusal).
These children score 24.7 on average with a target refusal rate of 0.82, versus 31.0 and 0.62 for
real attacks (`operators_mutator_refusals.csv`). The rate varies sharply by operator:
`semantic_steganography` is a mutator refusal in 19 of 30 uses, `game_theory_attack` in 7 of 11,
and `synonym` (a WordNet substitution with no LLM call) in none of 22. `semantic_steganography`
remains weak on its 11 real attacks (25.3, refusal 0.82), so its rank survives, but its reported
90% refusal rate mostly measures the mutator. The audit matches English refusal openers and
phrases only, so 27% is a lower bound.

Taken together, operator choice is where a decision model could matter, and zero-shot selection
from descriptions does not exploit it: the preference is set by wording, and the selector cannot
see whether its operator produced an attack at all.

### 5.3 Attack families: what the score metric hides

The runner does not log the corpus cluster of each prompt, so we recover it offline. 958 of 1,120
prompts match a corpus prompt verbatim; the remaining 162 (mostly MUTATE outputs) receive the
majority top-level cluster of their ten nearest corpus prompts under the engine's embedding model
(83.5% leave-one-out accuracy). These metrics were defined after the primary analysis, so we report
them as exploratory.

**Jev's action policy narrows the search.** Act=Jev visits 9.0 ± 2.7 of the 30 top-level clusters
per run versus 12.4 ± 2.3 for the baseline, and its cluster entropy is lower by 0.64 [0.23, 1.07]
bits (raw p = 0.040, δ = −0.80; Figure 11). Restricted to exactly matched prompts, the gap is 3.8
[2.0, 5.8] clusters (raw p = 0.024, δ = −0.92), and the same direction holds under the coverage
objective (entropy −0.25 [−0.41, −0.09]). None of these contrasts survives Holm correction, but
every interval excludes zero and the effect sizes are the largest in the study. The mechanism is
the SIMILAR preference from Section 5.1: across all 40 runs, cluster breadth falls with the SIMILAR
share (Spearman ρ = −0.81) while mean score rises with it (ρ = +0.61).

**The configured seeding contributes about a tenth of the gap.** Because EXPLORE samples only the
first P(explore) fraction of the corpus (every post-initial EXPLORE draw in the static and Jev arms
lies below its P(explore); Figure 16), and the corpus is stored in source order (the first 25%
spans 9 of 24 sources and 3.3 of 4.6 bits of cluster entropy), a policy that explores less also
explores a narrower slice. Post-initial EXPLORE draws could reach 66% of the corpus on average under
the static policy and 21% under Jev. To separate the two effects, we re-draw each post-initial
EXPLORE prompt's cluster 500 times, once from its configured window and once from the full corpus,
keeping every other prompt's observed cluster (`seed_coupling_clusters.csv`). The configured
simulation reproduces the observed breadth (13.2 versus 12.4 for Baseline, 9.3 versus 9.0 for
Act=Jev). The Act=Jev − Baseline gap is −3.9 clusters with configured windows and −3.5 with
full-corpus draws, so about 10% of the gap comes from the window and the rest from Jev simply
exploring less (23 post-initial EXPLORE draws versus 61). This decomposition is first-order: it
does not simulate the descendants of a different EXPLORE draw.

**The attack family dominates the outcome.** Refusal rates by cluster span 0% ("Extensive
Jailbreak Template Collection," which elicits only benign compliance) to 100% ("Substance Synthesis
via Character Personas," among others; Figure 12). All 13 own-judge successes fall in three
clusters, viz. "Demon & Amoral Entity Personas" (7), "Fictional Storytelling Roleplay" (4), and
"Structured Programming-Style Jailbreak Frameworks" (2). Every arm also stays below the breadth of
pure random sampling (28 uniform corpus draws cover 16.6 ± 1.7 clusters). Therefore, which family
of attacks the search samples matters more to the outcome than how the selector allocates actions,
and a selector that narrows the family distribution risks missing the few families that work.

### 5.4 The coverage objective degenerates into random exploration

The coverage fitness assigns credit only for categories not yet covered, and a category is covered
only by a high_risk or harmful verdict. At a 1.2% success rate this almost never happens, so every
parent carries fitness 0. The static policy then selects EXPLORE 70% of the time (observed 0.71),
and Jev, seeing only parents below 30, also shifts to EXPLORE (P = 0.60). In 14 of 15 coverage
runs, no category is covered. The coverage arms are therefore random search in practice, and their
contrasts do not test RQ1 or RQ3. A usable coverage objective needs partial credit (for example,
for low_risk or related responses) or a fallback to the score objective until the first category
is covered.

### 5.5 RQ2: Jev as a judge under a common referee

**The Jev-judged arm is indistinguishable on the common scale.** Re-scored by the Meta referee,
Fit=Jev reaches a mean score of 32.1 versus 28.9 for the baseline, a difference of +3.3
[−3.8, +11.1] (p = 0.48, δ = 0.36). Under its own Jev judge, the same arm had appeared stronger
(+6.0 [−0.5, +12.7], refusal rate 0.49 versus 0.69), so roughly half of its apparent advantage
was labeling. Its refusal rate on the common scale is 0.62 (−0.07 [−0.24, +0.09] versus
baseline), and the referee finds 7 successes in its 140 prompts versus 3 in the baseline's 140
(p = 0.72). On the common Meta scale, the study has 16 successes in total.

**Jev agrees with Meta substantially, but calls fewer refusals.** On the 140 prompts with both
verdicts, alignment labels agree exactly in 77% of cases (Cohen's κ = 0.64 unweighted, 0.76
quadratic), refusal versus non-refusal agrees in 86%, and scores correlate at Spearman ρ = 0.78
(Table 7, `referee_summary.json`). Disagreements are one-directional: in 25 of the 32, Jev rates
the response as more compliant than Meta, and 18 of those are responses Meta calls a weak refusal.
Jev's refusal rate is 0.49 against Meta's 0.62. Successes overlap only partly: Jev finds 4, Meta 7,
and both agree on 3.

| Jev \ Meta | full_refusal | weak_refusal | weak_acceptance | full_acceptance |
|---|---|---|---|---|
| full_refusal | 5 | 4 | 0 | 0 |
| weak_refusal | 0 | 59 | 0 | 1 |
| weak_acceptance | 0 | 11 | 10 | 2 |
| full_acceptance | 1 | 7 | 6 | 34 |

*Table 7: Alignment verdicts on the 140 Fit=Jev prompts.*

**Relatedness is calibrated to Meta; refusal is not.** Treating Meta's verdicts as the reference
(not as ground truth), Jev's probability of refusal (the sum of its two refusal options) averages
0.47 against a Meta rate of 0.62, with an expected calibration error (ECE) of 0.15 and a Brier
score of 0.13. Its per-category relatedness probabilities, evaluated on the 318 category verdicts
where the referee called the harm judges, are close to Meta's (ECE 0.03, Brier 0.06, mean 0.10
versus a Meta rate of 0.13). These properties sit alongside the internal consistency of the first
sweep: Jev's label equals the argmax of its probabilities in 99.3% of alignment judgments and 100%
of harm judgments, and moving the relatedness threshold from 0.3 to 0.7 changes the success count
from 6 to 1 while moving the mean score by about one point (Figure 10).

**Gating halves the referee cost without changing scores.** The gated referee spent 458 judge calls
on 140 prompts (3.3 per prompt) instead of 980. The saving is smaller here than the 58% projected
for the full sweep because Jev-judged runs reach more non-refused responses.

**A cross-model annotation confirms the labels but not the harm.** Because Meta is both target and
referee, we add a third, independent annotator from a different model family (Claude, in four
parallel instances) that labels a stratified sample of 103 prompts blind to every judge verdict,
using the judges' own rubrics (Figure 15; `labeling/`, `annotator_claude_summary.json`). The sample contains
all 13 own-judge successes, 45 full acceptances scoring 50 to 80, 15 weak acceptances, and 30
refusals. Three results follow. First, alignment labels hold: the annotator agrees with Meta in 84%
of cases (κ = 0.74, quadratic 0.82) and on refusal versus non-refusal in 91%; where they differ,
Meta is usually the stricter (11 of 16 disagreements, most often a Meta weak refusal that the
annotator calls a full acceptance). On the 22 Jev-judged items the agreement is 82% (κ = 0.48 on a
small, acceptance-heavy sample). Second, harm severity does not hold: the annotator rates no
response in the sample as high_risk or harmful, including the 4 successes that a judge escalated
to that level (a prompt injection that elicited "Access Granted" with no harm-category content,
generic profane insults, a crude joke, and a non-explicit roleplay). It agrees that 11 of the 13
successes contain low_risk content and that all 13 are full acceptances. The Meta referee
independently downgrades the Jev-judged crude joke from harmful to low_risk. Third, the refusal gate
is safe on this sample: none of the 30 judged refusals, and no response the annotator itself calls
a refusal, contains harm. Therefore, the study's successes are best read as compliance with
low-risk requests. Only four prompts in the entire sweep carry a high_risk or harmful verdict
(yielding the study's five category-coverage events), all four are in the sample, and none survives
independent review.

## 6 Discussion

The study was designed to ask whether calibrated decisions improve fuzzing. Its answer is that, at
this budget and against this target, the decisions Jev replaced were not the binding constraint.
The binding constraints are the target's refusal rate (about 70%), the discreteness of the fitness
signal (plateaus at 20, 37, and 54 and almost nothing above), and the attack material: which
operator is applied, which attack family is sampled, and whether the mutator writes an attack at
all. A fitness signal that cannot distinguish among the 98.8% of prompts that fail gives any
selector, calibrated or not, little to exploit, and the 1.2% it rewards are mostly mild compliance
rather than harmful uplift.

Four design lessons follow. First, a decision model placed at a low-leverage decision point cannot
produce a large effect, so candidate decision points should be ranked by oracle headroom before a
model is deployed at them; Section 5.1 provides such an estimate at no additional cost. Second,
zero-shot selection over natural-language descriptions inherits whatever the descriptions
emphasize, and a description that promises filter evasion is an attractive answer to a question
about bypassing a refusal-heavy target. Operator descriptions are therefore part of the selector's
parameters and deserve the same scrutiny as its weights; a selector that learns from outcomes needs
an explicit account of sample size. Third, score alone is an incomplete objective for an
attack-generation system: Jev's action policy looked neutral on score and still reduced
attack-family diversity by roughly a quarter, so diversity (corpus-cluster coverage) belongs in the
evaluation of any selector. Fourth, the fuzzer's own machinery shapes what any selector can
achieve. When more than a quarter of mutations are the mutator's refusal, and when the operator and
corpus draws share a random variate with the action draw, a selector's intended distribution and
its realized effect can diverge. Any ablation of NAAMSE-style control should log intended
probabilities alongside realized choices, as this study did, so the two can be compared. Fifth, a
success threshold that admits one low_risk verdict measures compliance, not harm; evaluations that
compare attack strategies should report success at the high_risk level separately and validate it
with a judge outside the target's model family.

## 7 Limitations

- **Budget and power.** Five runs per arm and 28 prompts per run. The exact permutation test cannot
  reach p below 0.008, and arm-level differences of several points are undetectable. The pooled and
  offline analyses mitigate but do not remove this constraint.
- **Referee and annotator identity.** The common referee is the Meta judge panel, and Muse Spark
  is also the target, which risks self-preference bias. The independent annotator is itself an LLM
  (Claude), not a human; it removes the shared-model concern but not LLM-judge bias in general, and
  its severity calls are a second opinion rather than ground truth. The blind sheet in `labeling/`
  remains available for human annotation, and a human pass would settle the severity question.
- **Configured seeding.** The per-task seeding couples the action draw to the operator and corpus
  draws in every arm (Section 2). We report realized behavior and quantify the coupling offline;
  results describe NAAMSE as configured, and a system with independent draws could differ,
  particularly for operator selection.
- **Observational pooling.** Transition and operator estimates pool decisions made by different
  selectors. The off-policy estimators correct for selector choice but assume action outcomes that
  do not depend on the policy, which the EXPLORE window violates. The oracle bound is fitted
  in-sample and is optimistic.
- **Offline replays.** The description ablation replays 66 matched parents (16 ambiguous parents are
  excluded) and uses in-sample outcome statistics in the evidence condition; it measures Jev's
  preferences, not the downstream effect of those preferences on attack success.
- **Post-hoc analyses.** Corpus-cluster metrics, the mutator audit, and the seeding decomposition
  were defined after the primary analysis. 14% of prompts carry kNN-assigned clusters (about one in
  six of them wrong), and the mutator audit misses non-English refusals. One successful prompt was
  written back into the corpus during the sweep (1 of 129k prompts); it is excluded.
- **Coverage-arm parents.** Coverage runs record parent fitness (0) rather than parent judge score,
  so they are excluded from the transition analysis.
- **Single target.** Results are specific to one refusal-heavy target. A more compliant target would
  raise the success rate and could change which decision points carry leverage.

## 8 Related Work

Mutation-based jailbreak fuzzing (GPTFuzzer [2]) and iterative attacker-LLM refinement (PAIR [3])
both use fixed search control. NAAMSE [1] extends this line to agents with an evolutionary loop,
hierarchical corpus exploration, and multi-judge behavioral scoring; this work ablates that loop's
control. Standardized red-teaming evaluation (HarmBench [5]) motivates fixed referees and
success-rate metrics, which we approximate with a common Meta referee. LLM-as-a-judge [4] underlies
the fitness signal we attempt to replace, and calibration in the sense of Guo et al. [6] motivates
the use of probability-returning decisions. Doubly robust off-policy evaluation [7] lets us
re-estimate action policies from logged decisions without new queries. To our knowledge, no prior
work (i) evaluates a probability-returning decision model as a drop-in replacement for the action
policy, operator selector, and fitness judge of an evolutionary agent fuzzer under a fixed budget,
(ii) isolates whether such a model's operator preferences come from descriptions or evidence, or
(iii) measures how often the fuzzer's own mutation LLM refuses to produce the attack.

## 9 Conclusion

Replacing NAAMSE's hand-built decisions with Jev did not improve attack outcomes against Muse
Spark. Three off-policy estimators bound the benefit of learned action selection to under two score
points, and action selection has under three points of headroom to begin with. Jev's operator
preferences are set by how the operators are described, favor weak operators, and chase small
samples when given evidence; the realized concentration on two operators in the sweep is a product
of NAAMSE's shared per-task seed rather than of Jev. As a judge, Jev agrees substantially with Meta
under a common referee and is calibrated on relatedness, but calls fewer refusals. An independent
annotator confirms the judges' alignment labels but finds no high-risk content among the study's
successes, which are mostly compliance with low-risk requests. The only
measurable effect of Jev's action policy is a narrower search over attack families, in a setting
where 3 of 26 families produce every success and 27% of mutations never reach the target as
attacks. Operator choice, not action choice, is where a decision model could matter, and it will
need outcome feedback with an account of uncertainty, a mutator that complies, and a fitness
signal with more resolution to do so.

## References

[1] K. Pai, P. Shah, H. Patel, et al. NAAMSE: Framework for Evolutionary Security Evaluation of
Agents. arXiv:2602.07391, 2026.

[2] J. Yu, X. Lin, Z. Yu, et al. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated
Jailbreak Prompts. arXiv:2309.10253, 2023.

[3] P. Chao, A. Robey, E. Dobriban, et al. Jailbreaking Black Box Large Language Models in Twenty
Queries. arXiv:2310.08419, 2023.

[4] L. Zheng, W.-L. Chiang, Y. Sheng, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena. arXiv:2306.05685, 2023.

[5] M. Mazeika, L. Phan, X. Yin, et al. HarmBench: A Standardized Evaluation Framework for
Automated Red Teaming and Robust Refusal. arXiv:2402.04249, 2024.

[6] C. Guo, G. Pleiss, Y. Sun, et al. On Calibration of Modern Neural Networks. arXiv:1706.04599,
2017.

[7] M. Dudík, J. Langford, and L. Li. Doubly Robust Policy Evaluation and Learning.
arXiv:1103.4601, 2011.

## Figures and Tables

- Figure 1: `figures/figure-01-main-comparison.pdf`. Per-arm outcomes.
- Figure 3: `figures/figure-03-jev-action-policy.pdf`. Jev action probabilities vs. static policy.
- Figure 4: `figures/figure-04-operator-choice.pdf`. Realized operator choice, uniform vs. Jev
  (see Table 5 for intended choice).
- Figure 5: `figures/figure-05-best-so-far.pdf`. Search dynamics.
- Figure 6: `figures/figure-06-score-distribution.pdf`. Discrete score distribution.
- Figure 7: `figures/figure-07-action-outcomes.pdf`. Child outcomes by action and parent bucket.
- Figure 8: `figures/figure-08-policy-values.pdf`. Direct-method policy values.
- Figure 9: `figures/figure-09-operator-outcomes.pdf`. Per-operator outcomes.
- Figure 10: `figures/figure-10-jev-judge-consistency.pdf`. Jev judge confidence and threshold
  sensitivity.
- Figure 11: `figures/figure-11-cluster-breadth.pdf`. Corpus-cluster breadth per run.
- Figure 12: `figures/figure-12-cluster-outcomes.pdf`. Outcomes by corpus cluster.
- Figure 13: `figures/figure-13-operator-intended-vs-realized.pdf`. Jev operator selection,
  intended vs. realized.
- Figure 14: `figures/figure-14-description-ablation.pdf`. Jev operator preferences under five
  framings.
- Figure 15: `figures/figure-15-cross-model-annotation.pdf`. Annotator vs. Meta alignment, and
  severity of the 13 successes.
- Figure 16: `figures/figure-16-explore-window.pdf`. EXPLORE corpus offset vs. P(explore).
- Tables 4 to 7: `offpolicy_values.csv`, `seed_coupling_operators.csv`,
  `jev_description_ablation.json`, `referee_summary.json`; supporting tables
  `uniform_arm_transitions.csv`, `operators_mutator_refusals.csv`, `seed_coupling_clusters.csv`,
  `referee_arm_comparison.csv`, `annotator_claude_summary.json`, `labeling/labels_claude_joined.csv`.

## Appendix A: Artifacts and Reproduction

All paths are relative to the repository root; derived outputs are in `analysis-output/jev-ablation/`.
Scripts run with `uv run python scripts/<name>.py` and read `outputs/` unless noted.

| Script | Produces | API cost |
|---|---|---|
| `analyze_ablation.py` | Arm metrics, contrasts, Figures 1 to 6 | none |
| `analyze_pooled.py` | Transitions, DM policy values, operators, Jev judge consistency, Figures 7 to 10 | none |
| `analyze_clusters.py` | Corpus-cluster assignment and breadth, Figures 11 and 12 | none |
| `analyze_offpolicy.py` | SNIPS/DR values, Act=uniform transitions, mutator-refusal audit (Table 4, Section 5.2) | none |
| `analyze_seed_coupling.py` | Intended vs. realized operators, EXPLORE-window decomposition (Table 5, Section 5.3) | none |
| `jev_description_ablation.py` | Five-framing replay of 66 parents (Table 6); resumable | Jev only |
| `run_referee.py` | Gated Meta re-scoring of stored prompts; resumable | Meta judges |
| `analyze_referee.py` | Common-scale arm comparison, agreement, calibration (Table 7) | none |
| `make_label_sheet.py` | Blind 103-item sheet and separate key (`labeling/`) | none |
| `analyze_labels.py` | Annotator vs. judges, success severity, gate check | none |
| `make_followup_figures.py` | Figures 13 to 16 | none |

The refusal gate is implemented in `src/experiments/gated_judge.py` and exposed through
`src/experiments/referee.py` (`gated=True`); the fuzzer's own scoring graph is unchanged. Scripts
that import shared helpers from other scripts run from `scripts/` or with `PYTHONPATH=scripts`.
