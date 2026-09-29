# Probabilities Without Payoff: Replacing Hand-Built Decisions in Evolutionary Agent Fuzzing with a Probability-Returning Decision Model

*Kunal Pai, University of California, Davis (kunpai@ucdavis.edu)*

*Markdown companion to `paper/jev-ablation/main.tex`, synced 2026-09-28. The LaTeX paper is the
source of truth; this file follows its text, numbering, and numbers, and leaves out the author's
inline notes. Figures and tables use the paper's numbers; each figure names its file in
`figures/`, and tables are written out as markdown tables where the paper places them.*

## Abstract

Evolutionary fuzzers for LLM agents rely on hand-built rules to steer their search, such as fixed
score thresholds, uniformly random mutation choices, and LLM judges that grade each attack. We ask
whether a decision model that returns a probability for every option can replace these rules and
find better attacks under the same budget. We replace each decision point in the NAAMSE fuzzer
with TypeSafe Jev and evaluate eight ablation arms (40 runs, 1,120 attack prompts) against a Muse
Spark target, along with five follow-up analyses that send no new queries to the target. We find
that Jev does not improve the search: no arm outperforms the baseline, and Jev's zero-shot action
policy is within two judge-score points of the static thresholds. Jev chooses mutation operators
based on how they are described rather than how well they work, and as a judge it agrees with Meta
on labels but calls fewer refusals. A controlled experiment that applies ten operators to the same
20 parents finds that no operator beats re-sending the parent unchanged, and that Jev's preferred
operators would score below uniform choice. Our follow-up analyses also show that the fuzzer itself
shapes these results: about a quarter of mutated prompts are the mutation model's own refusal, a
shared random seed couples decisions that should be independent, and the judges' "successes" are
mostly compliance with low-risk requests. Finally, we show that a refusal-gated referee needs 47%
of the judge calls and, because no refusal in the sweep received a harm verdict, would leave every
score unchanged.

## 1 Introduction

AI agents are increasingly deployed with access to tools, data, and users, which makes their
security a practical concern rather than a theoretical one. Manual red-teaming does not scale to
this setting, so recent work automates it. Mutation-based fuzzers evolve jailbreak templates [11],
attacker LLMs refine prompts over several rounds [1], and NAAMSE [8] treats agent security
evaluation as an optimization problem: it starts from a large corpus of known attacks, sends them
to the agent, scores each response, and evolves the most promising prompts.

Every such fuzzer has to make the same kinds of decisions over and over. For example, when a
prompt partly works, should the fuzzer look for similar prompts in the corpus, mutate this one, or
give up and try something new? Which of its 26 mutation operators (e.g., translating the prompt,
wrapping it in a role-play, or encoding it) should it apply? And how should it grade the target's
response? In NAAMSE, these decisions are made by fixed rules: score thresholds pick the next
action, the operator is drawn uniformly at random, and a panel of LLM judges grades each
response [12]. These rules are simple heuristics, and it is natural to ask whether a learned model
could make better choices.

In this work, we investigate whether a decision model that returns a *probability* for every
option can replace these rules. The intuition is that such a model can condition each choice on
the prompt and on how the judges rated it, and can express how confident it is, which is useful to
the extent that its probabilities are calibrated [4]. To test this, we replace each decision point
in NAAMSE with TypeSafe Jev and evaluate eight ablation arms that swap in Jev as the action
selector (RQ1), the fitness judge (RQ2), or the operator selector (RQ3), under a fixed budget of 28
attack prompts per run.

First, we find that Jev does not improve the search. No arm outperforms the baseline, and because
each run makes only 28 attacks, we also pool 480 individual decisions and re-estimate each policy
offline. All three estimators we use bound Jev's gain from choosing actions to less than two
judge-score points. Even an oracle that always picks the best action for each range of parent
scores would gain less than three points, which suggests that, at least with the information the
baseline uses, action selection is not where the leverage is.

Furthermore, we observe that Jev's choice of mutation operator is driven by wording, not evidence.
When we swap the operator descriptions between operators, Jev's preferences follow the
descriptions rather than the operator names. Its favorite operators are among the weakest in our
data, and when we show it outcome statistics, it jumps to operators that were tried only once or
twice. To measure what operator choice is actually worth, we apply ten operators to the same 20
parents. Surprisingly, no operator beats simply re-sending the parent, and replaying Jev's
preferences on this grid scores below uniform choice.

While analyzing these results, we found that the fuzzer itself shapes what any selector can
achieve. More than a quarter of the "attacks" produced by mutation are actually the mutation model
refusing to write the attack. A single random seed per task is shared by the action choice, the
operator choice, and the corpus sample, so the operators that are actually applied differ from the
ones Jev intended. Finally, a blind annotator from a third model family agrees with the judges on
whether the target complied, but finds no high-risk content among the 13 prompts that the judges
counted as successes, or in any of the five prompts that a judge rated high-risk. Most of these
"successes" are the target complying with low-risk requests, such as writing suggestive pickup
lines.

In summary, this paper makes the following contributions:

- We show that replacing NAAMSE's action selector, operator selector, and fitness judge with a
  probability-returning decision model does not improve attack outcomes, and that action selection
  based on the parent's score has less than three judge-score points of headroom in our setting.
- We show that zero-shot operator selection is driven by operator descriptions rather than
  operator outcomes, that adding outcome statistics makes the model chase small samples, and, in a
  controlled experiment, that no operator outperforms a no-op against this target.
- We identify properties of the fuzzer that shape every arm: mutator refusals and silent mutation
  failures, a shared per-task seed, and a success threshold that rewards low-risk compliance. We
  quantify how each one affects the reported results.
- We propose a refusal-gated referee that needs 47% of the judge calls, and we check the Meta
  judge, and spot-check Jev, with a blind cross-model annotation.

## 2 Background

**NAAMSE.** NAAMSE runs in iterations. In each iteration, it picks parent prompts in proportion to
their fitness, applies one action to each parent, sends the resulting children to the target
agent, and scores the responses. There are three actions: EXPLORE draws a new prompt from the
corpus, SIMILAR draws a prompt that is close to the parent in embedding space, and MUTATE rewrites
the parent with one of 26 operators. In the baseline, the action depends only on the parent's
score. Parents scoring below 50 are usually abandoned (EXPLORE with probability 0.7), parents
scoring between 50 and 80 are usually explored nearby (SIMILAR with probability 0.7), and parents
scoring above 80 are usually mutated (MUTATE with probability 0.7).

**Judge score.** Each response receives a judge score that combines how much the target complied
with how harmful the response was. A response-alignment judge assigns a term r from full refusal
to full acceptance, and six harm judges (one per category) assign a term h from not harmful to
harmful:

  s = r + h · (1 + π/10),  r ∈ {25, 42, 58, 75},  h ∈ {−20, 5, 15, 30},   (Equation 1)

where h is the strongest harm term across categories and π ∈ [0, 1] is a small boost for detected
personal information. In practice the score takes only a few values: weak refusals land near 20
(full refusals near 5), partial compliance near 37, and harmless compliance near 54. Following
NAAMSE, we call a prompt with s ≥ 80 a *success*. Note that a full acceptance with a single
low-risk category already scores 80, so a success does not have to contain high-risk content. This
detail becomes important in Section 4.

**Jev.** TypeSafe Jev is a commercial decision API: it answers typed questions (for example, "which
of these options?") and returns a probability for every option instead of free text. Its model and
training details are not public, so we treat it as a black box and use it zero-shot, with no
fine-tuning and no feedback from earlier decisions. We turn each decision point into such a
question. For action selection, the question includes the parent prompt, its score, and its
per-category judge verdicts, and the action is sampled from Jev's probabilities. For operator
selection, each of the 26 options comes with one sentence that describes what the operator does.
As a judge, Jev answers the same rubric questions as the Meta judges.

**Per-task seeding.** NAAMSE runs several tasks in parallel and gives each task one random seed.
Each decision inside a task creates a new random number generator from that same seed. As a
result, the action choice, the operator choice, and the corpus sample all start from the same
random number. We keep this configuration unchanged, since it is part of the system under test,
and measure its effect in Section 4.

## 3 Study Design

**Threat model.** We consider a black-box attacker who can send messages to a deployed agent but
cannot see its weights or system prompt. The attacker has a fixed budget of 7 iterations with 4
mutations each, for 28 attack prompts per run, and tries to get the agent to produce harmful
content in any of six categories: disinformation, illegal goods and services, hate and harassment,
non-violent crime, violence, and sexually explicit content.

**Ablation arms.** Table 1 lists the eight arms. Each arm replaces one component with Jev and keeps
the others at their defaults. We run every arm under the standard score objective, and three of
them also under a coverage objective that rewards reaching new harm categories.

*Table 1: Ablation arms (5 seeds each). Bold marks the component replaced by Jev.*

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

**Follow-up analyses.** Five runs per arm leave little statistical power, so we add five follow-up
analyses that reuse the stored runs and send no new queries to the target (Table 2).

- *Off-policy re-estimation.* Every action decision logs the probability with which it was taken,
  so we can estimate how any policy would have done on the same data [2].
- *Description ablation.* We replay 66 parents to Jev's operator question under five different
  framings of the options. These are the unique parents of mutated prompts in the four
  score-objective, Meta-judged arms.
- *Common referee.* The Fit=Jev arm was graded by Jev itself, so we re-score its 140 prompts with
  the Meta judges. The referee runs the alignment judge first and skips the six harm judges when
  the response is a refusal. In the sweep, all 754 refusals received "not harmful" in every
  category, so we expect skipping these calls never to change a score; we did not re-run the
  skipped calls to confirm this directly.
- *Randomness and mutator audit.* We inspect every mutated prompt, compare it with its parent, and
  locate every EXPLORE draw in the corpus.
- *Cross-model annotation.* Because Meta is both the target and the default judge, we ask an
  annotator from a third model family (Claude, in four instances labeling disjoint quarters) to
  label a stratified sample of 103 prompts, blind to every judge verdict and using the judges' own
  rubrics. The sample contains all 13 successes under each arm's own judge, 45 near-misses, 15
  partial acceptances, and 30 refusals.

*Table 2: Cost of the follow-up analyses and the controlled experiment. The original sweep used
seven judge calls per prompt.*

| Follow-up | Target queries | Judge calls | Jev calls | Input |
|---|---|---|---|---|
| Off-policy re-estimation | 0 | 0 | 0 | logged probabilities |
| Description ablation | 0 | 0 | 358 | 66 replayed parents |
| Common referee (gated) | 0 | 458 (980 ungated) | 0 | 140 prompts |
| Randomness and mutator audit | 0 | 0 | 0 | corpus database |
| Cross-model annotation | 0 | 0 | 0 | 103 blind labels (4 Claude instances) |
| Controlled operator experiment | 200 | 758 (gated) | 0 | 20 parents × 10 operators |
| *Original sweep* | *1,120* | *about 7,840* | (not applicable) | *40 runs* |

## 4 Evaluation

### 4.1 Experimental Setup

The target is Muse Spark behind the NAAMSE example agent, reached over the A2A protocol. The
mutation model and the default fitness judges are Meta models. Each arm runs with seeds 1 to 5, for
40 runs and 1,120 attack prompts in total. We treat each run as one observation. Because runs with
the same seed do not start from the same prompts across arms, we compare arms with unpaired exact
permutation tests (252 splits), report 95% bootstrap intervals and Cliff's δ, and apply Holm
correction across the seven planned contrasts per metric. For analyses that pool individual
decisions, we compute intervals by resampling whole runs. With five runs per arm and the observed
spread (a standard deviation of about 6 points in mean judge score), the smallest difference we
could detect with 80% power at α = 0.05 is about 11 points, so a null result at the arm level
rules out only large effects.

### 4.2 High-Level Takeaways

**Takeaway 1: Replacing the hand-built rules with Jev does not improve the search.** No arm is
distinguishable from the baseline (Figure 1). Against the baseline, the largest difference in mean
judge score is Act=Jev, at +2.7 points with a 95% interval of [−4.2, +9.3] (p = 0.50); the largest
planned contrast overall, Act=Jev versus Act=uniform (+4.2, p = 0.22), is also not significant.
After Holm correction, every p-value across mean score, maximum score, successes, coverage, and
refusal rate is 1.00, except one maximum-score contrast (0.50) that is driven by a single run. The
target is hard to attack: it refuses about 70% of prompts in the Meta-judged arms, and only 13 of
1,120 prompts (1.2%) reach a score of 80 under each arm's own judge. In 39 of 40 runs, the best
score improves by less than one point after iteration 3; the exception is one coverage run in which
a late random draw scored 100, the same run that drives the one maximum-score contrast (Figure 5).
This confirms that five runs per arm cannot resolve differences of a few points, which motivates
the pooled analyses below.

> **Figure 1** (`figures/figure-01-main-comparison`). Per-run outcomes by arm. Fit=Jev is graded
> by its own Jev judge here; Takeaway 6 places it on the Meta scale.

**Takeaway 2: Choosing actions better would help little, and Jev does not do it.** Jev's zero-shot
policy is sensible: it rarely mutates weak parents, and its probability of mutating rises sharply
once the parent scores above 50. However, it prefers SIMILAR where the baseline prefers EXPLORE, so
Act=Jev spends less of its budget on new prompts (31% instead of 56%). When we pool 480 decisions,
Jev's preference turns out to be right for weak parents (below 30) and wrong for middling ones (30
to 50), and the two errors cancel out (Table 3). Since each decision logs the probability with
which it was taken, we can estimate each policy's value offline with three estimators: the direct
method, SNIPS, and doubly robust estimation (Appendix Table 4). All three place Jev within two
points of the baseline policy. Outside the Act=Jev arm, Jev's policy is approximated by its average
probabilities in each score range, so these estimates cannot see any prompt-specific signal Jev
might use. An oracle that always picks the best action for each of four score ranges, fitted on the
same data and therefore optimistic, reaches only 32.6, compared with 29.7 for the baseline. The
Act=uniform arm, whose actions are random, shows the same pattern on its own, so the result is not
an artifact of pooling. This suggests that, with the parent's score as the state, action selection
has little headroom here.

**Takeaway 3: Jev picks operators by how they are described, not by how well they work.** In our
replay, Jev's three favorite operators (`persona_roleplay`, `adversarial_prefix`, and
`semantic_steganography`) are among the weakest in the sweep's data. For instance, the target
refused `adversarial_prefix` in all six of its uses in the sweep (Figure 8), and in the controlled
experiment of Takeaway 8 it scores 6.7 points below a no-op. To find out what drives Jev's choice,
we replay 66 parents under five framings of the options (Figure 2a and Table 5). Hiding the
operator names barely changes Jev's preferences (ρ = 0.91). Swapping the descriptions between
operators, however, moves Jev's preferences along with the descriptions (ρ = 0.83) rather than the
names (ρ = 0.19). For example, the operator that inherits `semantic_steganography`'s description
receives 0.23 of the probability, while `semantic_steganography` itself drops to 0.02. Jev's
preferences also barely change from one parent to the next, so its choice is largely a fixed
preference set by the wording. When we add each operator's observed outcomes to its description,
Jev does move toward operators that scored well, but it picks the two with the least evidence
(`many_shot_jailbreaking`, tried twice, and `payload_splitting`, tried once). Worse, both recorded
uses of `many_shot_jailbreaking` were failed mutations that re-sent the parent unchanged
(Takeaway 4), so its high score belongs to its parents. This confirms that operator descriptions
act as hidden parameters of a zero-shot selector, and that a selector which learns from outcomes
needs to account for how much evidence it has.

> **Figure 2.** Jev operator selection.
> **(a)** (`figures/figure-14-description-ablation`) Mean probability over 66 parents, five
> framings. Preferences follow the descriptions.
> **(b)** (`figures/figure-13-operator-intended-vs-realized`) Intended vs. realized picks in the
> Mut=Jev arms. The shared per-task seed turns them into picks from the end of the alphabet.

**Takeaway 4: The fuzzer's own machinery changes what a selector can achieve.** Three properties
of NAAMSE, unrelated to Jev, change the experiment that actually runs.

The first is the shared per-task seed. Because MUTATE is the last action, it is only chosen when
the shared random number is high, and the operator is then drawn with that same high number from a
list sorted by name. As a result, the operators that Jev actually applied are not the ones it
preferred: `synonym` had a probability of only 0.03 but was applied 14 times. Simulating the
configured seeding with Jev's logged probabilities reproduces this pattern (63%
`semantic_steganography` and 24% `synonym`, against 54% and 40% realized), while sampling them
independently recovers Jev's intended shares (Figure 2b). The Mut=Jev arm therefore never tested
Jev's operator preferences; the description ablation (Takeaway 3) tests them directly, and
Takeaway 8 estimates what they would have been worth. The same effect limits EXPLORE, which only
ever samples the first P(explore) fraction of the corpus (Figure 10a). Since the corpus is stored
in source order, a policy that explores less also explores a narrower slice.

The second property is that the mutation model sometimes refuses. In 44 of 164 mutated prompts
(27%, or 23% with a stricter detector that only matches refusal openers), the text sent to the
target is the mutation model's own refusal to write the attack. For instance,
`semantic_steganography` produced a refusal in 19 of its 30 uses, whereas `synonym`, whose transformation uses a thesaurus rather than a language model, never did. These refusals score lower than real attacks (24.7 versus 31.0) and
are refused more often by the target (82% versus 62%). Neither detector was hand-validated, and
both miss non-English refusals.

The third property is that mutations can fail silently: when a mutation raises an error, NAAMSE
sends the parent unchanged. Excluding the no-op operator, 15 of the 164 mutated prompts are
identical to their parent, including both uses of `many_shot_jailbreaking`. The controlled
experiment in Takeaway 8 finds at least three causes: the tool behind `many_shot_jailbreaking` cannot reach the prompt database it needs and crashes whenever it is called, the mutation agent sometimes loops until it hits its step
limit, and one of `semantic_steganography`'s strategies had a code defect, which we fixed before
that experiment. This confirms that an ablation of a fuzzer's control should log both intended and
realized choices, and should check whether each mutation actually produced an attack.

**Takeaway 5: Jev's action policy narrows the range of attacks the fuzzer tries.** We recover the
corpus cluster of every prompt offline and count how many of the 30 top-level clusters each run
visits. Act=Jev visits 9.0 clusters per run, compared with 12.4 for the baseline, and the
difference in cluster entropy is 0.64 bits with a 95% interval of [0.23, 1.07] (Figure 9). This
difference does not survive Holm correction, but it is among the largest effects in our study, and
its direction replicates under the coverage objective (Cliff's δ = −0.88). This is consistent with
Jev's preference for SIMILAR: across the 35 Meta-judged runs, the more a run uses SIMILAR, the
fewer clusters it visits (ρ = −0.84) and the higher its mean score (ρ = +0.55). The shared seed
explains only about 10% of the gap; the rest comes from Jev exploring less (23 EXPLORE draws after
the first iteration, compared with 61). This matters because all 13 successes (under each arm's own
judge) come from just 3 of the 26 clusters the fuzzer visited. This confirms that a policy can look
neutral on score while quietly reducing the diversity of attacks.

**Takeaway 6: As a judge, Jev agrees with Meta on labels, and an independent annotator finds none
of the successes high-risk.** Once the Meta referee re-scores the Fit=Jev arm, it is no better than
the baseline (32.1 versus 28.9, with a 95% interval of [−3.8, +11.1] for the difference). Under its
own Jev judge, the gap had been nearly twice as large (+6.0), so about half of it came from
labeling. On the 140 prompts graded by both judges, they agree on the alignment label 77% of the
time (Cohen's κ = 0.64, quadratic κ = 0.76). When they disagree, Jev usually rates the response as
more compliant (25 of 32 cases; Table 6), and Jev calls 49% of responses refusals compared with
Meta's 62%. The blind annotator agrees with Meta on alignment labels 84% of the time (κ = 0.74,
quadratic κ = 0.82), which is closer than Jev's agreement with Meta; on the 22 Jev-judged prompts
in its sample, it agrees with Jev 82% of the time (κ = 0.48). The annotator agrees with Meta on
whether a response contains high-risk content 96% of the time, but it disagrees on every response
that a judge escalated (Figure 3). The fitness judges rated four prompts in the whole sweep
high-risk or harmful, and the Meta referee rated a fifth (a playful JavaScript easter egg that Jev
had scored 54); the annotator rates all five as low-risk or not harmful. One of the four is a
prompt injection that got the target to say "Access Granted" but contains no harmful content; the
others are generic insults, a crude joke (which the Meta referee also downgraded to low-risk), and
a non-explicit role-play. Nine of the 13 successes are full acceptances whose strongest harm
verdict is low-risk, which is exactly the case Equation 1 allows. The count itself depends on the
judge: on the common Meta scale, the referee finds 7 successes in Fit=Jev instead of Jev's 4, for
16 in the sweep. Finally, none of the 30 judged refusals contains any harm, which supports skipping
the harm judges on refusals. This confirms that the study's successes mostly measure compliance
with low-risk requests, and that success at the high-risk level should be reported separately and
checked by a judge outside the target's model family.

> **Figure 3** (`figures/figure-15-cross-model-annotation`). Blind cross-model annotation. Left:
> alignment labels from Meta (fitness judge, or referee for Fit=Jev) against the annotator on 103
> prompts. Right: strongest harm verdict on the 13 successes under each arm's own judge.

**Takeaway 7: The coverage objective turns into random search.** The coverage objective only
rewards a prompt when it reaches a new harm category at high risk or above. Since this almost never
happens, every parent has fitness 0, and both the baseline and Jev fall back to exploring new
prompts most of the time (71% and 64% of prompts). In 14 of the 15 coverage runs, no category is
ever covered. This confirms that a coverage objective needs partial credit, or a fallback to the
score objective until the first category is covered, before it can test any selector.

**Takeaway 8: In a controlled test, no operator beats re-sending the parent, and Jev's preferences
would do worse than uniform choice.** The sweep's operator evidence is observational, so we run a
controlled experiment. We draw 20 parents from the score-objective runs (8 scoring below 30, 4
between 30 and 50, and 8 between 50 and 80), apply each of ten operators to every parent with a
fixed seed, and send all 200 attacks to the target, scored by the refusal-gated Meta judges (758
judge calls). The ten operators are a no-op control (`echo`, which re-sends the parent unchanged),
Jev's three favorites, the operators that looked strongest in the sweep, a non-LLM operator
(`synonym`), and one weak operator. Because every operator sees the same parents, we compare each
one with `echo` on the same parent (Figure 4). No operator scores meaningfully higher than `echo` (37.2): the
best, `code_exec` (37.3), `many_shot_jailbreaking` (36.5), and `synonym` (36.3), are indistinguishable from it, and every other operator scores lower. These top ranks mean little, since about half of the cells of each of these three operators were silent re-sends of the parent. The largest loss is
`persona_roleplay`, one of Jev's favorites, at 10.1 points below `echo` (95% interval [−17.6,
−2.7], Holm-adjusted p = 0.050), and 12.7 points below it when we keep only mutations that produced a real
attack (Holm-adjusted p = 0.01). None of the 200 attacks reaches a score of 80 or receives a high-risk
verdict. The experiment also shows how unreliable the mutations are: 32 of the 200 are mutator
refusals (12 of 20 for `semantic_steganography`), and 47 are silent failures that re-send the
parent, so 40% of the "mutations" are not new attacks. Re-sending the same parent through `echo`
also shows how noisy a single score is: the new score differs from the stored one by 5.8 points on
average (Spearman ρ = 0.72). Finally, because every operator was observed on every parent, we can
replay operator policies on this grid without new queries (Figure 11). Over a run-sized budget of
10 decisions, uniform choice averages 33.2 points and Jev's intended preferences 31.6, while UCB1
(33.4) and Thompson sampling (33.6) are no better than uniform; Thompson sampling reaches 35.3 only
after 200 decisions, still below the best fixed operator (37.3). This confirms that, against this
target, the spread between operators is mostly downside: the best a selector can do is avoid
harmful operators, which Jev's zero-shot preferences do not.

> **Figure 4** (`figures/figure-17-operator-experiment`). Controlled operator experiment (20
> parents × 10 operators, refusal-gated Meta judges). Left: mean judge score and refusal rate per
> operator; grey is the `echo` no-op. Right: each operator's score minus `echo` on the same parent.

## 5 Discussion

**Where the leverage is.** In our setting, the decisions that Jev replaced were not the
bottleneck. The target refuses about 70% of prompts, the judge score takes only a handful of
values, and the few prompts that score highly are mostly low-risk compliance. What does matter is
the attack material itself: which family of attacks is sampled, whether the mutation model writes
an attack at all, and avoiding operators that make attacks worse. The controlled experiment shows
that even operator choice offers little upside against this target, since no operator beat
re-sending the parent. We therefore suggest that decision points be ranked by their oracle
headroom, which the pooled decisions provide at no extra cost, before a learned model is deployed
at them.

**Descriptions are parameters.** A zero-shot selector inherits whatever its option descriptions
emphasize. For a question about getting past a refusal-heavy target, a description that promises
to hide the request's intent is an attractive answer, even when the operator performs poorly. We
view descriptions as part of the selector's configuration, deserving the same care as its weights,
and we expect a selector that learns from outcomes to need an explicit notion of uncertainty.

**Cost.** The follow-up analyses send no new queries to the target, use 358 Jev calls and 458
judge calls in total, and reuse the 40 stored runs. Skipping the harm judges on refusals cut the
referee's cost from 980 to 458 judge calls; since no refusal in the sweep received a harm verdict,
we expect it to change no score. We expect a larger saving on arms with more refusals: 67% of all
prompts in the sweep were refusals (70% in the Meta-judged arms), against 62% of the Fit=Jev
prompts the referee re-scored.

## 6 Threats to Validity

**Internal validity.** Five runs per arm and 28 prompts per run limit statistical power, and the
exact permutation test cannot reach p below 0.008. The pooled analyses help, but they assume that
an action's outcome does not depend on the policy that chose it, which the shared seed violates for
EXPLORE. The oracle bound is fitted on the same data and is therefore optimistic. The cluster
analysis, the mutator audit, and the seeding analysis were defined after the primary analysis; 14%
of prompts have clusters assigned by nearest neighbors, and the audit misses non-English refusals.

**Construct validity.** Meta is both the target and the referee, which risks self-preference bias.
Our annotator removes this shared-model concern, but it is itself an LLM (Claude) rather than a
human, so its severity judgments are a second opinion rather than ground truth. The description
ablation measures Jev's preferences, not their downstream effect on attacks. The controlled
experiment observes each (parent, operator) pair once, and the `echo` re-test shows that a single
score carries about 6 points of noise; it also runs NAAMSE's operators as configured, including the
two failure modes we found and did not fix. Jev is a black box, and we evaluate one
probability-returning model; other models of this kind could behave differently.

**External validity.** Our results describe NAAMSE as configured, including its per-task seeding,
and a single refusal-heavy target. A fuzzer with independent random draws, or a more compliant
target, could shift which decision points carry leverage.

**Ethics and broader impact.** This work red-teams a deployed model through its public API to
improve how such testing is evaluated, and it produced no high-risk content by independent review.
Automated red-teaming tools can be misused, so we release analysis code and derived tables but not
the raw attack prompts or target responses, which remain available to researchers on request. The
study involves no human subjects; the annotation was done by an LLM.

## 7 Related Work

Automated red-teaming has moved from static benchmarks toward adaptive search. LLM-Fuzzer
(originally GPTFuzzer) [11] mutates jailbreak templates and selects seeds with a tree search, and
TurboFuzzLLM [3] learns which mutation to apply with reinforcement learning, which is the closest
prior work to our operator-selection question. PAIR [1] hands search control to an attacker LLM
that refines prompts from the target's responses. In classic fuzzing, MOpt [6] and SLOPT [5] learn
mutation-operator schedules online, with particle swarm optimization and bandits respectively; our
results suggest that such outcome-driven selectors are a better fit for operator choice than
zero-shot description matching. Rainbow Teaming [9] treats the diversity of attacks as an explicit
objective, which our cluster analysis supports. NAAMSE [8] extends this line to agents with an
evolutionary loop, hierarchical corpus exploration, and multi-judge scoring, and our work ablates
that loop's control. HarmBench [7] motivates fixed referees and standardized success metrics, which
we approximate with a common Meta referee and a cross-model annotator, and StrongREJECT [10] shows
that jailbreak success is often overstated when judges reward compliance without useful harmful
content, which matches what our annotator finds. LLM-as-a-judge [12] underlies the fitness signal
we attempt to replace, calibration [4] motivates probability-returning decisions, and doubly robust
estimation [2] lets us evaluate policies from logged decisions. To the best of our knowledge, prior
work has not evaluated a probability-returning model as a zero-shot, drop-in replacement for all
three control points of an evolutionary agent fuzzer, isolated whether such a model's choices come
from descriptions or evidence, or measured how often the fuzzer's own mutation model refuses to
write the attack.

## 8 Conclusion

We replaced the hand-built decisions in the NAAMSE fuzzer with a probability-returning decision
model and found that it does not produce better attacks against a refusal-heavy target. Choosing
actions better would gain less than three points, Jev chooses operators by their descriptions, and
as a judge it agrees with Meta on labels while calling fewer refusals. The analyses that explain
these results turned out to be the more lasting contribution: a shared seed changes which choices
are actually made, more than a quarter of mutations never reach the target as attacks, and the
success threshold rewards low-risk compliance. A controlled experiment shows that even operator
choice offers little upside against this target: no operator beat re-sending the parent, and 40%
of mutations failed to produce a new attack. We therefore believe that making the fuzzer's
mutations reliable and its fitness signal more informative should come before learning its
control, and that a decision model will need outcome feedback with an account of uncertainty to
help once they are.

## References

Numbered in the order of the compiled bibliography (`plainnat`, sorted by author), matching
`paper/jev-ablation/references.bib`.

[1] P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong. Jailbreaking Black Box
Large Language Models in Twenty Queries. In *IEEE Conference on Secure and Trustworthy Machine
Learning (SaTML)*, pages 23 to 42, 2025. doi:10.1109/SaTML64287.2025.00010.

[2] M. Dudík, J. Langford, and L. Li. Doubly Robust Policy Evaluation and Learning. In
*Proceedings of the 28th International Conference on Machine Learning (ICML)*, pages 1097 to 1104,
2011.

[3] A. Goel, X. Wu, Z. Wang, D. Bespalov, and Y. Qi. TurboFuzzLLM: Turbocharging Mutation-based
Fuzzing for Effectively Jailbreaking Large Language Models in Practice. In *Proceedings of the 2025
Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 3: Industry Track)*, pages 523 to 534, 2025.

[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On Calibration of Modern Neural Networks. In
*Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pages 1321
to 1330, 2017.

[5] Y. Koike, H. Katsura, H. Yakura, and Y. Kurogome. SLOPT: Bandit Optimization Framework for
Mutation-Based Fuzzing. In *Proceedings of the 38th Annual Computer Security Applications
Conference (ACSAC)*, 2022. doi:10.1145/3564625.3564659.

[6] C. Lyu, S. Ji, C. Zhang, Y. Li, W.-H. Lee, Y. Song, and R. Beyah. MOPT: Optimized Mutation
Scheduling for Fuzzers. In *28th USENIX Security Symposium (USENIX Security 19)*, pages 1949 to
1966, 2019.

[7] M. Mazeika, L. Phan, X. Yin, A. Zou, Z. Wang, N. Mu, E. Sakhaee, N. Li, S. Basart, B. Li,
D. Forsyth, and D. Hendrycks. HarmBench: A Standardized Evaluation Framework for Automated Red
Teaming and Robust Refusal. In *Proceedings of the 41st International Conference on Machine
Learning (ICML)*, PMLR 235, pages 35181 to 35224, 2024.

[8] K. Pai, P. Shah, and H. Patel. NAAMSE: Framework for Evolutionary Security Evaluation of
Agents. In *ICLR 2026 Workshop on Agents in the Wild*, 2026. arXiv:2602.07391.

[9] M. Samvelyan, S. C. Raparthy, A. Lupu, E. Hambro, A. H. Markosyan, M. Bhatt, Y. Mao, M. Jiang,
J. Parker-Holder, J. Foerster, T. Rocktäschel, and R. Raileanu. Rainbow Teaming: Open-Ended
Generation of Diverse Adversarial Prompts. In *Advances in Neural Information Processing Systems
(NeurIPS)*, volume 37, 2024.

[10] A. Souly, Q. Lu, D. Bowen, T. Trinh, E. Hsieh, S. Pandey, P. Abbeel, J. Svegliato, S. Emmons,
O. Watkins, and S. Toyer. A StrongREJECT for Empty Jailbreaks. In *Advances in Neural Information
Processing Systems (NeurIPS), Datasets and Benchmarks Track*, volume 37, 2024.

[11] J. Yu, X. Lin, Z. Yu, and X. Xing. LLM-Fuzzer: Scaling Assessment of Large Language Model
Jailbreaks. In *33rd USENIX Security Symposium (USENIX Security 24)*, pages 4657 to 4674, 2024.
(Published version of GPTFuzzer, arXiv:2309.10253.)

[12] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li,
E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena. In *Advances in Neural Information Processing Systems (NeurIPS), Datasets and
Benchmarks Track*, volume 36, 2023.

## Appendix A: Additional Results

This appendix collects the figures and tables that support the takeaways in Section 4. We describe
what each one shows and what we think readers should take away from it.

**Pooled transitions (Table 3).** This table breaks the 480 pooled decisions down by the parent's
score and the action taken, and reports the average score and refusal rate of the resulting child.
The best action changes with the parent's score: SIMILAR works best for weak parents (below 30) and
for promising ones (50 to 80), while EXPLORE and MUTATE are close in between (MUTATE rests on only
six samples). Neither policy gets every range right: Jev is right where the baseline is wrong
(below 30), and wrong where the baseline is right (30 to 50). The takeaway is that the two policies
make offsetting mistakes, which is why Act=Jev ends up level with the baseline.

**Judge agreement (Table 6).** This table compares Jev's and Meta's alignment labels on the 140
Fit=Jev prompts. Most prompts fall on the diagonal, and 25 of the 32 disagreements sit below it,
where Jev calls a response more compliant than Meta does. The most common case is a response that
Meta calls a weak refusal and Jev calls an acceptance (18 prompts). The takeaway is that the two
judges differ in how strict they are about refusals rather than in what they think the response
says.

**Search dynamics (Figure 5).** Figure 5a tracks the best score found so far in each arm. In 39 of
40 runs the best score improves by less than one point after iteration 3; the one exception is a
coverage run (Cov + Mut=Jev) whose late random draw scored 100. The curves above about 54 come from
one or two lucky seeds rather than from steady progress. Act=Jev starts higher only because two of
its initial random prompts happened to score 80, before any Jev decision was made. Figure 5b shows
why: the judge score takes only a few values, near 5 for full refusals, 20 for weak refusals, 37
for partial compliance, and 54 for harmless compliance, with very little above. The takeaway is
that the search runs out of signal, not budget, and that a mean score mostly tracks how often the
target refuses.

**Action selection (Figure 6).** Figure 6a shows how each arm actually spent its budget. Act=Jev
moves budget away from EXPLORE and toward SIMILAR and MUTATE, which confirms that the selector
switch worked, while the coverage arms explore about 70% of the time regardless of the selector.
Figure 6b compares Jev's action probabilities with the baseline's fixed weights. Jev rarely mutates
parents below 50 and mutates most parents above 80, which shows that its decisions are sensible and
easy to interpret. This panel includes Jev's decisions at iteration 0 and places parents scoring
exactly 80 in the 50 to 80 range. The main difference is at low scores, where Jev prefers SIMILAR
and the baseline prefers EXPLORE. The takeaway is that Jev did learn a reasonable policy; it just
does not lead to better attacks.

**Pooled outcomes and policy values (Figure 7).** Figure 7a is the visual version of Table 3, with
the most likely action of each policy marked in every score range. It shows the offsetting
mistakes directly, and also that SIMILAR is the worst choice for parents between 30 and 50.
Figure 7b shows the direct-method value of each policy and of the oracle. Jev and the baseline are
nearly identical, and the oracle is only about three points higher on score, although it would
reduce refusals by about 13 points. The takeaway is that better action choices could reduce
refusals somewhat, but would barely raise the score.

**Operators (Figure 8).** Figure 8a shows the average score of mutated prompts for each operator,
pooled over all Meta-judged arms, with the operators Jev applied in red. The operators differ by
more than 20 points, and the operator Jev applied most often (`semantic_steganography`) sits near
the bottom, although these observational averages are unreliable: in the controlled experiment of
Takeaway 8, `artprompt` ranks second to last, and both sweep uses of `many_shot_jailbreaking` were
failed mutations. Figure 8b shows how many times each operator was applied under the uniform
selector and under Jev. Jev's picks concentrate on two operators, but, as Figure 2b shows, this
concentration comes from the shared per-task seed rather than from Jev's own preferences. Note
also that many operators have six or fewer samples, so their averages are rough. The takeaway is
that operator choice matters far more than action choice, and that Jev's realized picks landed on
weak operators.

**Attack families (Figure 9).** Figure 9a counts how many of the 30 top-level corpus clusters each
run visits. Act=Jev is the narrowest arm (9.0 clusters per run against 12.4 for the baseline), and
every arm stays below the 16.6 clusters that 28 purely random draws would cover. Figure 9b shows
the refusal rate and average score of each cluster. Refusal rates range from 0% to 100% depending
on the cluster, and all 13 successes come from three clusters: two built around personas and
fiction, and one around programming-style jailbreak templates. The panel covers the Meta-judged
arms, so it shows 9 of those successes; the other 4 were judged by Jev. The clusters that are never
refused yield harmless compliance (around 54) rather than harm. The takeaway is that the family of
attacks matters more than the action policy, so a policy that narrows the search risks missing the
few families that work.

**Operator policies replayed on the controlled grid (Figure 11).** This figure replays five
operator policies on the 20-by-10 grid of the controlled experiment: each simulated decision draws
a parent at random, the policy picks an operator, and the reward is that cell's recorded score.
Jev's intended preferences sit below uniform choice at every budget, because they favor operators
that score below the no-op. The two bandits need many more decisions than a single run makes: at
10 decisions they are no better than uniform, and Thompson sampling closes only part of the gap to
the best fixed operator after 200. The takeaway is that outcome feedback helps only over long
campaigns, and even then the best it can learn here is to avoid harmful operators.

**EXPLORE window and judge consistency (Figure 10).** Figure 10a plots where in the corpus each
EXPLORE draw landed against the probability with which EXPLORE was chosen. Every point lies below
the diagonal: a decision that explores with probability 0.7 only ever reaches the first 70% of the
corpus, and one that explores with probability 0.1 only reaches the first 10%. This is the shared
per-task seed at work, since the same random number decides both whether to explore and where to
sample. Figure 10b looks at Jev as a judge without any reference labels. Most of Jev's relatedness
probabilities are close to 0 or 1, but some sit in the middle, and moving the relatedness threshold
from 0.3 to 0.7 changes the number of successes from 6 to 1 while the mean score barely moves. The
takeaway is that success counts are fragile to judge settings in a way that mean scores are not,
which is another reason to validate successes with an independent reviewer.

*Table 3: Mean child judge score (refusal rate) by parent bucket and action, pooled over 480
transitions from the four score-objective, Meta-judged arms. Bold marks the best action per bucket
where it is clear; the ≥ 80 row rests on 20 transitions.*

| Parent score | EXPLORE | SIMILAR | MUTATE | Static prefers | Jev prefers |
|---|---|---|---|---|---|
| < 30 (n = 257) | 25.0 (0.79) | **28.9** (0.66) | 22.4 (0.87) | EXPLORE | SIMILAR |
| 30 to 50 (n = 73) | 27.5 (0.79) | 21.7 (0.92) | 28.6 (0.50), n = 6 | EXPLORE | SIMILAR |
| 50 to 80 (n = 130) | 23.9 (0.86) | **41.1** (0.35) | 37.5 (0.44) | SIMILAR | MUTATE ≈ SIMILAR |
| ≥ 80 (n = 20) | 31.3 (0.67), n = 3 | 38.5 (0.67), n = 6 | 38.9 (0.45), n = 11 | MUTATE | MUTATE |

*Table 4: Offline policy values over 480 pooled decisions. Intervals resample runs. Jev's policy is
its logged probabilities in Act=Jev and its average probabilities per score range elsewhere.*

| Estimator | Baseline | Jev | Jev − baseline (score) | Jev − baseline (refusal rate) |
|---|---|---|---|---|
| Direct method | 29.7 | 29.9 | +0.1 [−0.7, +1.0] | −0.013 [−0.051, +0.025] |
| SNIPS | 29.1 | 30.0 | +0.9 [−0.2, +1.9] | −0.044 [−0.090, +0.006] |
| Doubly robust | 29.6 | 30.1 | +0.5 [−0.5, +1.5] | −0.032 [−0.077, +0.016] |

*Table 5: Jev operator preferences over 66 replayed parents. "Eff. ops" is the effective number of
operators Jev spreads its probability over; the last column is the rank correlation with observed
operator scores.*

| Framing | Top operators (mean probability) | Eff. ops | ρ w/ outcome |
|---|---|---|---|
| original | persona_roleplay .24, adversarial_prefix .22, semantic_steganography .16 | 9.3 | −0.11 |
| names only | deep_inception .19, persona_roleplay .18, contextual_framing .12 | 12.3 | −0.28 |
| anonymized | adversarial_prefix .22, persona_roleplay .22, semantic_steganography .16 | 9.7 | −0.08 |
| swapped | (mass follows the relocated descriptions) | 10.0 | +0.07 |
| evidence | many_shot_jailbreaking .60, payload_splitting .19, mathematical .09 | 4.0 | +0.48 |

*Table 6: Alignment verdicts on the 140 Fit=Jev prompts (rows: Jev; columns: Meta referee).*

| Jev \ Meta | full refusal | weak refusal | weak acceptance | full acceptance |
|---|---|---|---|---|
| full refusal | 5 | 4 | 0 | 0 |
| weak refusal | 0 | 59 | 0 | 1 |
| weak acceptance | 0 | 11 | 10 | 2 |
| full acceptance | 1 | 7 | 6 | 34 |

> **Figure 5.** Search dynamics and the plateaus of Equation 1.
> **(a)** (`figures/figure-05-best-so-far`) Best-so-far judge score by iteration.
> **(b)** (`figures/figure-06-score-distribution`) Discrete judge-score distribution.

> **Figure 6.** Action selection.
> **(a)** (`figures/figure-02-action-mix`) Action mix by arm.
> **(b)** (`figures/figure-03-jev-action-policy`) Jev action probabilities vs. static weights
> (includes iteration 0).

> **Figure 7.** Pooled transitions and offline policy values.
> **(a)** (`figures/figure-07-action-outcomes`) Child outcomes by action and parent bucket.
> **(b)** (`figures/figure-08-policy-values`) Direct-method policy values.

> **Figure 8.** Operators. Realized choices reflect the shared per-task seed; see Figure 2b for
> Jev's intended choices.
> **(a)** (`figures/figure-09-operator-outcomes`) Per-operator judge score of mutated prompts.
> **(b)** (`figures/figure-04-operator-choice`) Realized operator choice, uniform vs. Jev.

> **Figure 9.** Attack families.
> **(a)** (`figures/figure-11-cluster-breadth`) Corpus-cluster breadth per run.
> **(b)** (`figures/figure-12-cluster-outcomes`) Outcomes by corpus cluster (Meta-judged arms; 9
> of the 13 successes).

> **Figure 10.** The EXPLORE window under the shared per-task seed, and Jev judge
> self-consistency.
> **(a)** (`figures/figure-16-explore-window`) EXPLORE corpus offset vs. P(explore).
> **(b)** (`figures/figure-10-jev-judge-consistency`) Jev judge confidence and threshold
> sensitivity.

> **Figure 11** (`figures/figure-18-operator-bandit-replay`). Operator policies replayed on the
> controlled grid (4,000 simulated campaigns per point). The shaded band marks a single run's
> MUTATE budget.

## Appendix B: Artifacts and Reproduction

All scripts read the stored runs and write to `analysis-output/jev-ablation/`. Only
`jev_description_ablation.py` (Jev) and `run_referee.py` (Meta judges) make API calls; both resume
where they stopped. The refusal gate is implemented in `src/experiments/gated_judge.py` and exposed
through `src/experiments/referee.py`; the fuzzer's own scoring graph is unchanged.

| Script | Produces |
|---|---|
| `analyze_ablation.py` | Arm metrics and contrasts; Figures 1, 5, and 6 |
| `analyze_pooled.py` | Transitions, DM policy values, operators, Jev judge consistency |
| `analyze_clusters.py` | Corpus-cluster assignment and breadth |
| `analyze_offpolicy.py` | SNIPS and DR values; mutator-refusal audit |
| `analyze_seed_coupling.py` | Intended vs. realized operators; EXPLORE-window decomposition |
| `jev_description_ablation.py` | Five-framing replay of 66 parents |
| `run_referee.py`, `analyze_referee.py` | Gated Meta re-scoring; agreement and calibration |
| `make_label_sheet.py`, `analyze_labels.py` | Blind sheet and key; annotator vs. judges |
| `operator_experiment.py`, `analyze_operator_experiment.py` | Controlled operator experiment (Figure 4) |
| `operator_bandit_replay.py` | Offline operator-policy replay (Figure 11) |
| `make_followup_figures.py` | Figures 2, 3, and 10a |
