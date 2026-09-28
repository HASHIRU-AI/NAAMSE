# Probabilities Without Payoff: Replacing Hand-Built Decisions in Evolutionary Agent Fuzzing with a Calibrated Decision Model

*Draft mini-paper, 2026-09-27. All numbers are computed by `scripts/analyze_ablation.py`
and `scripts/analyze_pooled.py` from the 40 runs in `outputs/`. Figures are in `figures/`.*

## Abstract

Evolutionary fuzzers for LLM agents (NAAMSE among them) steer their search with hand-built
control: fixed score thresholds decide between exploring, exploiting, and mutating;
mutation operators are drawn uniformly at random; and a panel of LLM judges supplies the
fitness signal. Whether a probability-returning decision model can replace these components
has not been tested under a fixed attack budget. We replace each decision
point in NAAMSE with TypeSafe Jev (a model that returns a probability for every answer
option) and evaluate eight ablation arms (40 runs, 1,120 attack prompts) against a Muse Spark
target, under both a score objective and a harm-category coverage objective. No arm
outperforms the baseline: every Holm-adjusted p-value on outcome metrics is at least 0.50.
Pooling 480 individual decisions across arms bounds the effect of Jev's action policy to
+0.1 [−0.7, +1.0] score points relative to the static thresholds. Even an in-sample oracle
policy gains only +2.9 points. Jev's operator selection collapses onto 2 of 26 operators,
and its most frequent choice (`semantic_steganography`, refused 90% of the time) ranks 19th
of 25. Therefore, under a small budget against a refusal-heavy target, decision calibration
is not the bottleneck: operator choice is where the leverage lies, and zero-shot selection
from descriptions misses it. Analysis code and derived per-prompt tables are released with
NAAMSE.

## 1 Introduction

Automated red-teaming of LLMs and agents has moved from static benchmarks toward adaptive
search. Mutation-based fuzzers evolve jailbreak templates (GPTFuzzer [2]), attacker LLMs
refine prompts iteratively (PAIR [3]), and NAAMSE [1] reframes agent security evaluation as
feedback-driven optimization over a corpus of seed attacks. Each of these systems
hard-codes the control of its search. In NAAMSE, three components carry that control:
(i) a *score-threshold policy* that chooses among EXPLORE (sample a new corpus prompt),
SIMILAR (sample a neighbor of the parent), and MUTATE (apply an operator to the parent);
(ii) a *uniform draw* over 26 mutation operators; and (iii) a panel of LLM judges
(LLM-as-a-judge [4]) whose verdicts form the fitness signal.

These components are heuristics, and heuristics invite replacement. A decision model that
returns calibrated probabilities (in the sense of Guo et al. [6]) is a natural candidate:
it can condition each choice on the parent prompt and its judge verdicts, it exposes its
uncertainty, and it can score responses without free-text generation. We hypothesize that
such a model improves the search along three axes, viz. action selection (RQ1), fitness
judging (RQ2), and operator selection (RQ3).

To address this question empirically, we replace each decision point in NAAMSE with TypeSafe
Jev behind command-line ablation switches (defaults reproduce the unmodified system), and we
run a full factorial slice of the design against a deployed-class target. The result is
negative, and the negative result is informative. In this paper, we make the following
contributions:

- **A controlled ablation of fuzzer control:** eight arms isolating Jev as action selector,
  operator selector, and fitness judge, each under a score and a coverage objective
  (40 runs, 1,120 attack prompts, identical budgets).
- **A bounded null result for learned action selection:** pooling decisions across arms
  bounds Jev's policy to within one score point of the static thresholds, and an oracle
  bound shows that action selection itself has less than three points of headroom.
- **A failure mode for zero-shot operator selection:** Jev concentrates 33 of 35 picks on
  two operators, one of which is among the weakest in the pool, while operator means span
  more than 20 points.
- **Two design pathologies surfaced by the study:** a coverage objective with no gradient
  before its first success (degenerating into random exploration) and seeding that does not
  pair runs across arms.

## 2 Background

**NAAMSE.** Each iteration selects parent prompts in proportion to fitness, applies one
action per parent, sends the children to the target over A2A, and scores the responses. The
final score combines a response-alignment term (25, 42, 58, or 75 for full refusal, weak
refusal, weak acceptance, and full acceptance) with the maximum harm term across six
categories (−20, 5, 15, or 30 for not harmful through harmful), boosted by at most 10% for
PII. The score is therefore discrete in practice: refusals land near 20, partial compliance
near 37, benign compliance near 54, and any score of 80 or above requires a judge to find
harm. We call a prompt with score ≥ 80 a *success*.

**Static action policy.** For parent scores below 50, the weights over
(EXPLORE, SIMILAR, MUTATE) are (0.7, 0.2, 0.1); for 50 to 80, (0.1, 0.7, 0.2); for 80 to
100, (0.1, 0.2, 0.7).

**Jev.** Jev answers typed questions (choice, yes/no) with a probability per option. We
pose each decision point as a choice question: the `state` carries the parent prompt, its
score, and its per-category judge verdicts, and the action is sampled from the returned
probabilities (temperature 1.0). For operator selection, the options are the 26 operators,
each described by one curated sentence, with no run history. As a judge, Jev answers the
same category guidance as the Meta judges, and a category counts as related when
P(related) ≥ 0.5.

## 3 Threat Model and Setup

**Adversary.** A black-box attacker with query access to a deployed agent over A2A, no
access to weights or system prompt, and a fixed budget of 7 iterations × 4 mutations
(28 attack prompts per run). The attacker's goal is to elicit responses judged harmful in
any of six categories (disinformation, illegal goods and services, hate and harassment,
non-violent crime, violence, sexually explicit content).

**Target and judges.** The target is Muse Spark behind the NAAMSE example A2A agent. The
mutation LLM and the default fitness judges are Meta models. The Fit=Jev arm uses Jev
judges instead.

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

**Metrics.** Mean and maximum judge score per run, number of successes, harm-category
coverage (categories with a related response judged high_risk or harmful), refusal rate,
and action shares.

**Statistics.** The unit of analysis is the run. Same-seed runs do not share starting
prompts across arms (each selector consumes the random stream differently), so all
contrasts are unpaired: exact two-sided permutation tests on the difference of run means
(252 splits), 95% bootstrap intervals, Cliff's δ, and Holm correction across seven planned
contrasts per metric. For the pooled analyses (Section 5.2 onward), intervals come from a
cluster bootstrap that resamples whole runs.

## 4 Arm-Level Results

**No arm is distinguishable from the baseline.** Figure 1 shows per-run outcomes. The
largest mean-score contrast is Act=Jev − Baseline = +2.7 [−4.2, +9.3] (p = 0.50, δ = 0.28).
Across mean score, maximum score, successes, coverage, and refusal rate, every
Holm-adjusted p-value is 1.00, except maximum score for Cov + Mut=Jev versus its baseline
(0.50, driven by a single run reaching 100).

**Successes are rare and clustered.** Only 13 of 1,120 prompts (1.2%) reach score ≥ 80.
They come from 6 of 40 runs, and three runs (Baseline seed 2, Act=Jev seed 4, Fit=Jev
seed 2) contain 10 of them. Coverage is zero in 35 of 40 runs. The target refuses about
70% of attack prompts (Baseline refusal rate 0.69 ± 0.15), and the modal non-refused
outcome is benign compliance (score ≈ 54; Figure 6).

**The search stalls rather than running out of budget.** Best-so-far curves flatten by
iteration 2 to 3 in every arm (Figure 5). Act=Jev starts higher at iteration 0 only because
two initial EXPLORE samples scored 80, before any Jev decision could matter.

Taken together, five runs per arm cannot resolve differences smaller than several score
points at this success rate. The remaining sections recover power by pooling individual
decisions.

## 5 Where the Leverage Is (and Is Not)

### 5.1 RQ1: Jev learns a sensible policy that does not pay off

**Jev's policy is monotone in parent score.** Figure 3 plots Jev's mean action
probabilities by parent-score bucket against the static weights. P(MUTATE) rises from 0.14
(parents below 30) to 0.80 (parents at 80 or above), and P(EXPLORE) falls from 0.37 to
0.04. The direction matches the static policy, but Jev prefers SIMILAR where the static
policy prefers EXPLORE (parents below 50). As a result, Act=Jev shifts its budget from
EXPLORE (0.56 → 0.31 of prompts) toward SIMILAR (0.29 → 0.44) and MUTATE (0.16 → 0.26).

**Pooled transitions show where Jev is right and where it is wrong.** We pool 480
transitions (iteration ≥ 1) from the four score-objective, Meta-judged arms and measure the
child produced by each action per parent bucket (Figure 7, Table 2).

| Parent score | EXPLORE | SIMILAR | MUTATE | Static prefers | Jev prefers |
|---|---|---|---|---|---|
| < 30 (n = 257) | 25.0 (0.79) | **28.9** (0.66) | 22.4 (0.87) | EXPLORE | **SIMILAR** |
| 30 to 50 (n = 73) | **27.5** (0.79) | 21.7 (0.92) | 28.6 (0.50), n = 6 | **EXPLORE** | SIMILAR |
| 50 to 80 (n = 130) | 23.9 (0.86) | **41.1** (0.35) | 37.5 (0.44) | **SIMILAR** | MUTATE ≈ SIMILAR |

*Table 2: Mean child score (refusal rate) by parent bucket and action. The ≥ 80 bucket
(n = 20) is omitted as too small.*

In the largest bucket, Jev's preference is correct and the static policy's is not. In the
30 to 50 bucket the reverse holds. The two errors cancel.

**The policy effect is bounded, and so is the headroom.** Using a direct-method estimate
(expected child outcome under each policy's action probabilities, weighted by the pooled
bucket distribution), Jev and the static policy are indistinguishable: Jev − static =
+0.1 [−0.7, +1.0] score points and −1.3 [−5.1, +2.5] points of refusal rate (Figure 8). This
interval is roughly seven times narrower than the arm-level contrast. An in-sample oracle
that always selects the best action per bucket reaches 32.6 [30.0, 35.7] versus 29.7 for
the static policy: action selection can buy at most about three score points here, although
it can cut refusals from 0.67 to 0.54. Therefore, calibrated action selection does not
matter much in this setting because action selection itself does not matter much.

### 5.2 RQ3: Zero-shot operator selection collapses onto a weak operator

**Jev concentrates its picks.** Across both objectives, Jev chose MUTATE operators 35 times
and used only four operators: `semantic_steganography` (19) and `synonym` (14) account for
33 picks. Uniform selection used 12 distinct operators in 38 picks (Figure 4). All ten runs with Jev
operator selection draw from the same two operators (the only exceptions are one `unicode`
and one `persona_roleplay` pick), whatever their parents, which indicates a fixed preference induced by the operator descriptions rather than a parent-conditioned
decision.

**The preferred operator is among the weakest.** Pooling all 164 MUTATE prompts from the
seven Meta-judged arms (Figure 9), `semantic_steganography` scores 23.3 [20.3, 26.5] with a
90% refusal rate (30 prompts, 17 runs), ranking 19th of 25 operators. Operators Jev never
selected perform better: `artprompt` reaches 41.5 [36.8, 50.5] (n = 6, one success), and
`many_shot_jailbreaking` reaches 58.5 (n = 2, one success). Jev's second choice, `synonym`,
is mid-pack (34.9, n = 22). Consequently, Jev's MUTATE prompts score no better than
uniform's (28.0 versus 28.7; Mann-Whitney p = 0.88), and neither selector converts a MUTATE
step into a success.

**Operator choice is where the leverage lies.** Refusal rates range from 0% to 100% across
operators and operator means span more than 20 points, an order of magnitude more than the
action-selection headroom in Section 5.1. The failure is not that operator choice is
irrelevant: it is that description-only, zero-shot selection chose badly.

### 5.3 The coverage objective degenerates into random exploration

The coverage fitness assigns credit only for categories not yet covered, and a category
is covered only by a high_risk or harmful verdict. At a 1.2% success rate this almost never
happens, so every parent carries fitness 0. The static policy then selects EXPLORE 70% of
the time (observed 0.71), and Jev, seeing only parents below 30, also shifts to EXPLORE
(P = 0.60). In 14 of 15 coverage runs, no category is covered. The coverage arms are
therefore random search in practice, and their contrasts do not test RQ1 or RQ3. A usable
coverage objective needs partial credit (for example, for low_risk or related responses) or
a fallback to the score objective until the first category is covered.

### 5.4 RQ2: What can be said about Jev as a judge without a referee

The Fit=Jev arm is scored by Jev, while all other arms are scored by Meta, so its higher
mean score (34.9) and lower refusal rate (0.49 versus 0.69) cannot be attributed to the
search: they may reflect labeling differences alone. Without a common referee, we restrict
RQ2 to internal properties (Figure 10):

- **Labels follow probabilities.** Jev's label equals the argmax of its probabilities in
  99.3% of alignment judgments and 100% of harm judgments. Rescoring from stored
  probabilities reproduces every stored score exactly.
- **Harm verdicts are near-certain.** The median maximum harm probability is 1.00. For
  alignment, the median is 0.89, and 14% of calls have a maximum below 0.6.
- **Successes are threshold-sensitive; means are not.** Only 4.3% of relatedness
  probabilities fall within ±0.1 of the 0.5 threshold, yet 48% of prompts have at least one
  category between 0.3 and 0.7. Moving the threshold from 0.3 to 0.7 changes the success
  count from 6 to 1 (4 at the deployed 0.5), while the mean score moves by about one point.

Whether these probabilities are calibrated against any external standard remains open.

## 6 Discussion

The study was designed to ask whether calibrated decisions improve fuzzing. Its answer is
that, at this budget and against this target, the decisions Jev replaced were not the
binding constraint. The binding constraints are the target's refusal rate (about 70%), the
discreteness of the fitness signal (a fitness landscape with plateaus at 20, 37, and 54
and almost nothing above), and the quality of the operators applied. A fitness signal that
cannot distinguish among the 98.8% of prompts that fail gives any selector, calibrated or
not, little to exploit.

Two design lessons follow. First, a decision model placed at a low-leverage decision point
cannot produce a large effect, so candidate decision points should be ranked by
oracle headroom before a model is deployed at them (Section 5.1 provides such an estimate
at no additional cost). Second, zero-shot selection over natural-language descriptions
inherits whatever the descriptions emphasize. Both of Jev's favorites are word-substitution
operators: `semantic_steganography` "hides intent by swapping flagged words for innocent
alternatives," and `synonym` "substitutes key words with WordNet alternatives." Against a
refusal-heavy target, a description that promises filter evasion is an attractive answer to
the question Jev is asked, yet the first of these operators is refused 90% of the time.

## 7 Limitations

- **Budget and power.** Five runs per arm and 28 prompts per run. The exact permutation
  test cannot reach p below 0.008, and arm-level differences of several points are
  undetectable. The pooled analyses mitigate but do not remove this constraint.
- **No common referee.** All outcomes come from the fitness judge of each arm. RQ2 is
  reduced to internal consistency, and Muse Spark serves both as target and (through the
  Meta judges) as evaluator, which risks self-preference bias.
- **Observational pooling.** Transition and operator estimates pool decisions made by
  different selectors. Within-bucket differences in parent quality, and the operator mix
  within MUTATE, can confound them. The oracle bound is fitted in-sample and is optimistic.
- **Coverage-arm parents.** Coverage runs record parent fitness (0) rather than parent
  judge score, so they are excluded from the transition analysis.
- **Single target.** Results are specific to one refusal-heavy target. A more compliant
  target would raise the success rate and could change which decision points carry
  leverage.

## 8 Related Work

Mutation-based jailbreak fuzzing (GPTFuzzer [2]) and iterative attacker-LLM refinement
(PAIR [3]) both use fixed search control. NAAMSE [1] extends this line to agents with an
evolutionary loop, hierarchical corpus exploration, and multi-judge behavioral scoring;
this work ablates that loop's control. Standardized red-teaming evaluation (HarmBench [5])
motivates fixed referees and success-rate metrics, which our study lacks. LLM-as-a-judge [4]
underlies the fitness signal we attempt to replace, and calibration in the sense of Guo et
al. [6] motivates the use of probability-returning decisions. To our knowledge, no prior
work evaluates a probability-returning decision model as a drop-in replacement for the
action policy, operator selector, and fitness judge of an evolutionary agent fuzzer under a
fixed budget.

## 9 Conclusion

Replacing NAAMSE's hand-built decisions with Jev did not improve attack outcomes against
Muse Spark. Pooled evidence bounds the benefit of learned action selection to about one
score point, shows that action selection has little headroom to begin with, and exposes a
collapse in zero-shot operator selection onto an operator that is refused 90% of the time.
Operator choice, not action choice, is where a decision model could matter, and it will
need parent-conditioned context (and a fitness signal with more resolution) to do so.

## References

[1] K. Pai, P. Shah, H. Patel, et al. NAAMSE: Framework for Evolutionary Security Evaluation
of Agents. arXiv:2602.07391, 2026.

[2] J. Yu, X. Lin, Z. Yu, et al. GPTFUZZER: Red Teaming Large Language Models with
Auto-Generated Jailbreak Prompts. arXiv:2309.10253, 2023.

[3] P. Chao, A. Robey, E. Dobriban, et al. Jailbreaking Black Box Large Language Models in
Twenty Queries. arXiv:2310.08419, 2023.

[4] L. Zheng, W.-L. Chiang, Y. Sheng, et al. Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena. arXiv:2306.05685, 2023.

[5] M. Mazeika, L. Phan, X. Yin, et al. HarmBench: A Standardized Evaluation Framework for
Automated Red Teaming and Robust Refusal. arXiv:2402.04249, 2024.

[6] C. Guo, G. Pleiss, Y. Sun, et al. On Calibration of Modern Neural Networks.
arXiv:1706.04599, 2017.

## Figures

- Figure 1: `figures/figure-01-main-comparison.pdf`. Per-arm outcomes.
- Figure 3: `figures/figure-03-jev-action-policy.pdf`. Jev action probabilities vs. static
  policy.
- Figure 4: `figures/figure-04-operator-choice.pdf`. Operator choice, uniform vs. Jev.
- Figure 5: `figures/figure-05-best-so-far.pdf`. Search dynamics.
- Figure 6: `figures/figure-06-score-distribution.pdf`. Discrete score distribution.
- Figure 7: `figures/figure-07-action-outcomes.pdf`. Child outcomes by action and parent
  bucket.
- Figure 8: `figures/figure-08-policy-values.pdf`. Offline policy values.
- Figure 9: `figures/figure-09-operator-outcomes.pdf`. Per-operator outcomes.
- Figure 10: `figures/figure-10-jev-judge-consistency.pdf`. Jev judge confidence and
  threshold sensitivity.
