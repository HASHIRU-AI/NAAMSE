# Jev Ablation Sweep: Analysis Report

Date: 2026-09-27 (revised 2026-09-28) · Source: `outputs/` (40 runs) · Script: `scripts/analyze_ablation.py`

## Revisions (2026-09-28)

This report was written before the follow-up analyses. The source of truth is now
`paper/jev-ablation/main.tex` and the data files in this directory. The following earlier claims
are superseded; directly contradicted statements in the body below have been corrected in place
and marked *[revised]*.

1. **"Jev's operator choice collapses to about 2 of 26 operators"** (finding 4, finding 8, "What
   changed"). *Superseded by the seeding audit* (Figure 13, `seed_coupling_operators.csv`). The
   concentration on `semantic_steganography` (19) and `synonym` (14) is Jev's *realized* picks,
   not its preference. NAAMSE gives each task one seed, and each decision creates a fresh
   generator from it, so the action draw, the operator draw, and the EXPLORE corpus draw start
   from the same random number. MUTATE, the last action, is only chosen when that number is high,
   and the operator is then drawn with the same number from a name-sorted list, so picks land on
   late-alphabet operators. Jev's intended preferences are `adversarial_prefix` 0.26,
   `persona_roleplay` 0.19, `semantic_steganography` 0.19; `synonym` had 0.03 but was applied 14
   times. The seeding is the system's configured behavior, not a bug to fix.
2. **"Jev's favorite operator is among the worst performers"** (finding 8). *Refined.* Jev's
   intended favorites are also weak (`adversarial_prefix` was refused in all 6 of its uses), and a
   description ablation (Figure 14, `jev_description_ablation.json`) shows that Jev's preferences
   follow the operator descriptions (ρ = 0.83 when descriptions are swapped) rather than the
   names (ρ = 0.19). With outcome statistics added, Jev jumps to the operators with the least
   evidence (`many_shot_jailbreaking`, tried twice; `payload_splitting`, tried once).
3. **Operator refusal rates measured the operator** (finding 8). *Superseded by the mutator
   audit* (`operators_mutator_refusals.csv`). 44 of 164 MUTATE prompts (27%) are the mutation
   LLM's own refusal; `semantic_steganography` produced one in 19 of its 30 uses, so its 90%
   refusal rate mostly measures the mutator. A stricter detector gives 38 of 164 (23%); neither detector was hand-validated, and both miss non-English refusals.
4. **"RQ2 can't be evaluated" / "No referee was run"** (header, finding 6, finding 9, caveat 3,
   recommendation 1). *Superseded by the refusal-gated Meta referee* (`referee_summary.json`,
   `referee_arm_comparison.csv`). On the Meta scale, Fit=Jev is +3.3 [−3.8, +11.1] versus the
   baseline (own-judge gap +6.0). Jev and Meta agree on 77% of alignment labels (κ = 0.64
   unweighted, 0.76 quadratic); Jev calls 49% of responses refusals versus Meta's 62%. The gated
   referee used 458 judge calls instead of 980, with scores that we expect to be unchanged (no refusal in the sweep received a harm verdict; the skipped calls were not re-run).
5. **Success severity** (finding 2). *Refined by a blind cross-model annotation* (Claude, 103
   items; `annotator_claude_summary.json`, Figure 15). The annotator agrees with Meta on alignment
   (κ = 0.74) but rates none of the responses high_risk or harmful, including all 13 successes;
   9 of the 13 successes are full acceptances whose strongest harm verdict is low_risk. A success
   (score ≥ 80) does not require high-risk content.
6. **Cluster narrowing** (finding 10). *Holds.* The EXPLORE window created by the shared seed
   explains only about 10% of the Act=Jev versus Baseline gap (Figure 16,
   `seed_coupling_clusters.csv`); the rest comes from Jev exploring less.
7. **Offline policy bound** (finding 7). *Extended.* SNIPS and DR agree with the direct method:
   Jev − static is +0.1 (DM), +0.9 [−0.2, +1.9] (SNIPS), and +0.5 [−0.5, +1.5] (DR) score points
   (`offpolicy_values.csv`). These estimators assume an action's outcome does not depend on the
   policy, which the EXPLORE window violates.
8. **"Make seeds pair runs" / "Investigate RQ3's collapse"** (recommendations 4 and 5).
   *Withdrawn.* The per-task seeding is part of the system under test and stays unchanged; its
   effect is measured offline instead. The RQ3 investigation was done (item 2 above).
9. **"RQ3 is where the leverage is: operators differ by more than 20 points"** ("What changed").
   *Superseded by a controlled operator experiment* (main.tex Takeaway 8, Figures 17 and 18;
   `operator_experiment_summary.json`, `operator_experiment_per_operator.csv`,
   `operator_experiment_contrasts.csv`, `operator_bandit_replay.csv`). We applied 10 operators to
   the same 20 parents (200 attacks, 758 refusal-gated Meta judge calls) and compared each operator
   with the `echo` no-op on the same parent. No operator beats `echo` (37.2): `code_exec` (37.3),
   `many_shot_jailbreaking` (36.5), and `synonym` (36.3) are indistinguishable from it, and every
   other operator scores lower. The largest loss is `persona_roleplay`, one of Jev's favorites, at
   10.1 points below `echo` (95% interval [−17.6, −2.7], Holm p = 0.05). None of the 200 attacks
   reaches 80 or gets a high-risk verdict. About 40% of the mutations are not real attacks: 32 are
   mutator refusals and 47 are silent failures that re-send the parent. Replaying operator policies
   on this grid over 10 decisions, Jev's intended preferences (31.6) score below uniform choice
   (33.2). The 20-point spread in the sweep was observational and mostly downside, so the earlier
   advice to focus on operator choice no longer holds; the best a selector can do here is avoid
   harmful operators.

## Question
Does replacing NAAMSE's hand-built decision points with Jev probabilities change fuzzing
outcomes against the Muse Spark target?
- **RQ1**: Jev action selection (EXPLORE/SIMILAR/MUTATE) vs. static thresholds and uniform.
- **RQ2**: Jev as the fitness judge.
- **RQ3**: Jev mutation-operator choice vs. uniform.
- Each is run under the **score** and **coverage** objectives.

**Design.** 8 arms × 5 seeds, 7 iterations × 4 mutations = 28 prompts per run, 1,120 prompts
total. Mutation LLM: Meta. The fitness judge is Meta for every arm except Fit=Jev. No referee
was run during the sweep; a refusal-gated Meta referee was run afterwards on the Fit=Jev prompts
*[revised; see Revisions item 4]*.
**Unit of analysis.** One run. Runs are unpaired across arms (see caveat 2).

## Key findings

1. **No contrast shows a detectable effect on outcome metrics.** This covers mean score, max
   score, number of successes and coverage. Every Holm-adjusted p = 1.00, except max score for
   Cov+Mut=Jev (0.50). The 95% CIs are wide and include zero for every mean-score
   contrast; the largest is Act=Jev − Baseline = +2.7 [−4.2, +9.3]. With 5 runs per arm, the
   sweep can detect only very large effects. This is a null result under low power, **not
   evidence that the arms are equivalent**.

2. **Successes are too rare to compare arms.** Only 13 of 1,120 prompts (1.2%) reach score ≥ 80.
   *[revised]* This threshold does not require high-risk content: a full acceptance with a
   single low_risk verdict already scores 80, and 9 of the 13 successes are exactly that
   (Revisions item 5). They come from 6 of the 40 runs, and 10 of
   the 13 are in three runs: Baseline seed 2 (3), Act=Jev seed 4 (4, in one lineage) and
   Fit=Jev seed 2 (3). Coverage is 0 of 6 harm categories in 35 of 40 runs. The target refuses
   about 70% of prompts, and the most common score for non-refused prompts is ≈54 (complied,
   nothing harmful).

3. **RQ1 (mechanism): Jev learns a reasonable, score-dependent policy, but it didn't change
   outcomes.** Figure 3 shows P(mutate) rising from 0.14 for parents scoring below 30 to 0.80
   for parents scoring 80 or above, while P(explore) falls from 0.37 to 0.04. This is the same
   direction as the static thresholds, but Jev prefers SIMILAR much more at low scores (0.48
   vs. the static 0.20). Overall, Act=Jev shifts the action mix from EXPLORE (0.56 → 0.31)
   toward SIMILAR (0.29 → 0.44) and MUTATE (0.16 → 0.26). Its MUTATE prompts score higher on
   average (35.4 vs. 29.1 in the baseline) because they are applied to better parents. None of
   this produced a significant difference in outcomes.

4. **RQ3: Jev's realized operator picks concentrate on 2 of 26 operators, with no gain.**
   *[revised: the concentration comes from the shared per-task seed, not from Jev's
   preference; see Revisions item 1 and Figure 13.]* In both objectives, 33 of Jev's 35
   realized MUTATE picks were `semantic_steganography_mutation` (19) or `synonym_mutation` (14). Uniform selection used 12 different operators across 38 picks.
   Jev's MUTATE prompts score no better (mean 28.0 vs. 28.7; prompt-level Mann–Whitney
   p = 0.88), and neither selector produced any success from a MUTATE step. Jev's intended
   probabilities are nearly the same whatever the parent prompt, which fits a zero-shot
   preference based on the operator descriptions rather than on the parent; the description
   ablation confirms this (Revisions item 2).

5. **The coverage objective is degenerate at this success rate.** Coverage fitness is 0 until
   some category reaches high_risk, and that almost never happens. So every parent scores 0,
   and the static policy picks EXPLORE 70% of the time (observed 0.71). Jev also shifts to
   EXPLORE (P = 0.60; all 140 parents scored below 30). In practice the coverage arms are
   random search, so the "under coverage" contrasts don't test RQ1 or RQ3.

6. **RQ2 couldn't be evaluated from the sweep alone** *[revised: the common referee now places
   Fit=Jev at +3.3 [−3.8, +11.1] on the Meta scale; see Revisions item 4]*. The Fit=Jev arm is scored by Jev, while every
   other arm is scored by Meta. Its higher mean (34.9) and lower refusal rate (0.49 vs. 0.69)
   may just mean Jev labels responses differently from Meta; they don't show the fuzzer found
   better prompts. `referee_judge` was `none` in all runs.

## Pooled analyses (no new runs; `scripts/analyze_pooled.py`)

The arm-level tests use 5 runs per arm. These analyses pool individual decisions across arms
and resample whole runs to get the uncertainty right.

7. **Transitions (RQ1): Jev's action preference helps at low parent scores, hurts at
   mid-range scores, and nets to zero.** Data: 480 transitions (iteration ≥ 1) from the 20
   score-objective, Meta-judged runs.
   - *Parents scoring below 30* (54% of decisions): SIMILAR produces the best children, with
     mean 28.9 [25.9, 31.6] and 66% refusal, vs. EXPLORE at 25.0 with 79% refusal and MUTATE
     at 22.4 with 87% refusal. Jev's preferred action here is SIMILAR (P = 0.63), which is
     right; the static rule's is EXPLORE (0.7), which is wrong.
   - *Parents scoring 30–50:* SIMILAR is the worst action (21.7, 92% refusal, n = 25), and
     Jev still prefers it (0.66).
   - *Parents scoring 50–80:* SIMILAR and MUTATE are both good (41.1 and 37.5). Static
     prefers SIMILAR; Jev splits between the two.
   - *Offline policy value:* Jev − static = **+0.1 [−0.7, +1.0] score points** and
     **−1.3 [−5.1, +2.5] points of refusal rate**. This rules out any large effect of
     Jev's action policy. The interval is about 7× tighter than the arm-level contrast.
   - *Headroom:* even an optimistic policy that picks the best action per bucket, measured
     on this same data, gains only **+2.9 score points** over static (32.6 vs. 29.7). It does
     cut refusals by about 13 points (0.54 vs. 0.67). Action choice has little leverage on
     score, and Jev captures about 1 of the 13 available refusal points.

8. **Operators (RQ3): the operator Jev applied most often is among the worst performers.**
   *[revised: "applied" means realized under the shared seed; Jev's intended favorites are
   `adversarial_prefix`, `persona_roleplay`, and `semantic_steganography`, also weak. See
   Revisions items 1 to 3.]* `semantic_steganography_mutation` accounts for 19 of Jev's 35
   realized picks. Across 30 prompts in
   17 runs, it scores 23.3 [20.3, 26.5], with 90% refused. That's 19th of 25 operators and
   below every other operator with n ≥ 5 except two (deep_inception and
   adversarial_prefix). Operators Jev never picked do better:
   - artprompt: 41.5 [36.8, 50.5], n = 6, 1 success
   - dual_response_divider: 36.9, n = 7
   - many_shot_jailbreaking: 58.5, n = 2, 1 success

   Jev's other realized pick, `synonym_mutation`, is mid-pack (34.9, n = 22), although Jev gave
   it only 0.03 probability. Operator outcomes
   vary widely (refusal from 0% to 100%), so operator choice *does* matter. The problem is
   that zero-shot selection from descriptions prefers weak operators. Note that 19 of the 30
   `semantic_steganography` prompts are the mutator's own refusal *[revised; Revisions item 3]*.

9. **Jev judge (RQ2, self-consistency only): its labels come straight from its
   probabilities, it is highly confident, and its success count depends on the relatedness
   threshold.**
   - Labels equal the highest-probability option in 99.3% of alignment and 100% of harm
     judgments.
   - Harm probabilities are near-certain: the median highest probability is 1.00. For
     alignment the median is 0.89, and 14% of calls have a highest probability below 0.6.
   - Relatedness probabilities are mostly decisive: only 4% fall within ±0.1 of the 0.5
     threshold. But 48% of prompts have at least one category between 0.3 and 0.7.
   - Moving the threshold from 0.3 to 0.7 changes the number of successes from 6 to 1 (4 at
     the deployed 0.5), while the mean score moves by only about 1 point. Successes stay in
     1–2 runs throughout.
   - *[revised]* The common referee now supplies Meta labels on the same 140 prompts: 77%
     exact agreement, quadratic κ = 0.76, and Jev calls fewer refusals (49% vs. 62%).

## Corpus-cluster coverage (exploratory; `scripts/analyze_clusters.py`)

The runner didn't log which corpus cluster each prompt came from, so clusters were recovered
offline:
- **958 of 1,120 prompts** match a corpus prompt verbatim and get their exact cluster.
- **The other 162** get the majority top-level cluster of their 10 nearest corpus prompts,
  using MiniLM embeddings. Leave-one-out accuracy on exactly matched prompts is 83.5%.
- **Planned vs. post hoc:** these metrics were added after seeing the outcome results.

10. **Jev's action policy trades cluster breadth for depth, and the depth doesn't pay.**
    - *Breadth:* Act=Jev visits 9.0 ± 2.7 of the 30 top-level clusters per run vs. 12.4 ± 2.3
      for the baseline, and its cluster entropy is lower by 0.64 [0.23, 1.07] bits (raw
      p = 0.040, δ = −0.80).
    - *Exact-match prompts only:* −3.8 [−5.8, −2.0] clusters (raw p = 0.024, δ = −0.92).
    - *Under the coverage objective:* the same direction (entropy −0.25 [−0.41, −0.09],
      raw p = 0.024).
    - *Holm correction:* nothing survives (smallest adjusted p = 0.17). But the direction
      agrees across three metrics and both objectives, and every 95% CI excludes zero. These
      are the largest effects in the study. *[revised]* The EXPLORE window created by the
      shared seed explains only about 10% of the Act=Jev vs. Baseline gap; the finding holds
      (Revisions item 6).
    - *Mechanism:* across all 40 runs, breadth falls as the SIMILAR share rises (Spearman
      ρ = −0.81), while the SIMILAR share rises with mean score (ρ = +0.61). Jev's preference
      for SIMILAR buys a small, non-significant gain in mean score at a clear cost in attack
      diversity.
    - *Operator selection* (Mut=Jev) has no effect on breadth (−0.8 [−3.8, +2.2]), as
      expected, since mutations keep the parent's cluster.

11. **Which cluster an attack comes from matters more than any selector.**
    - *Refusal varies enormously by cluster:* from 0% ("Extensive Jailbreak Template
      Collection", n = 13, benign compliance at ≈54) to 100% (e.g. "Substance Synthesis via
      Character Personas", "Toxic Content via Controversial Personas").
    - *Successes are concentrated:* all 13 come from 3 of the 26 clusters visited, viz.
      "Demon & Amoral Entity Personas" (7), "Fictional Storytelling Roleplay" (4) and
      "Structured Programming-Style Jailbreak Frameworks" (2). 12 of the 13 have exact
      cluster labels.
    - *Search is narrow overall:* every arm visits fewer clusters than pure random sampling
      would (28 uniform corpus draws cover 16.6 ± 1.7). The coverage-objective arms come
      closest (13.4), in line with their near-random exploration.

## Main caveats
1. **Low power.** n = 5 runs per arm. The exact permutation test's minimum two-sided p is
   2/252 ≈ 0.008, before Holm correction over 7 contrasts.
2. **Seeds don't pair runs.** Same-seed runs in different arms start from different
   iteration-0 prompts (Act=Jev vs. Baseline share 3 of 4; Act=uniform shares 1 of 4), because
   each selector consumes the RNG differently. All tests are unpaired. This is a property of the
   configured per-task seeding, which we keep unchanged.
3. **No referee in the sweep**, so arm-level scores come from the fitness judge (Meta, or Jev
   for Fit=Jev). *[revised]* The refusal-gated Meta referee now puts Fit=Jev on the Meta scale
   (Revisions item 4). Meta is both the target's model family and the judge, which risks
   self-preference bias; the blind Claude annotation removes this concern for alignment labels
   (κ = 0.74) but is itself an LLM, not a human.
4. **The score is a discrete composite**, so means mostly reflect refusal rates. See
   Figure 6.
5. **The search stalls, it doesn't run out of budget.** Best-so-far curves flatten by
   iteration 2–3 in every arm (Figure 5). More iterations alone are unlikely to help; the
   search needs to find more successes. Act=Jev's head start at iteration 0 comes from lucky
   initial prompts, not from Jev.

6. **Caveats specific to the pooled analyses:**
   - *Transitions are observational.* Which parent gets which action depends on the arm's
     selector, so the action-by-bucket means may be confounded by parent score within a
     bucket and by the operator mix inside MUTATE.
   - *The policy values are direct-method estimates* and assume those means carry over
     between policies.
   - *Parent scores for coverage arms aren't recorded* (history stores fitness, which is 0),
     so those arms are excluded from the transition analysis.
   - *Operator means ignore parent quality.* Act=Jev applied MUTATE to better parents.

7. **Caveats specific to the cluster analysis:**
   - *Exploratory:* the cluster metrics weren't planned; treat raw p-values as descriptive.
   - *Approximate labels:* 14% of prompts carry a kNN-assigned cluster, about 1 in 6 of them
     wrong.
   - *Corpus write-back:* one success from the Act=Jev seed-4 run was written back into the
     corpus during the sweep (`NAAMSE_mutation`). It is excluded here, and its effect on
     later runs is negligible (1 of 129k prompts).
   - *Logging fixed:* the runner now logs `cluster_info` for future runs.

## What changed in our understanding
- The current setup has too few successes to answer RQ1 or RQ3 on outcome metrics. The
  limiting factor is the target's refusal rate, not the selector.
- Jev's action policy (RQ1) is interpretable and behaves differently from the static policy.
  That is a result in itself (what Jev chooses), even though it didn't change outcomes.
- *[revised]* Jev's mutation selection (RQ3) is driven by the operator descriptions, not by
  outcomes, and its intended favorites are weak operators. The narrow realized pattern in the
  sweep comes from the shared per-task seed, not from Jev (Revisions items 1 and 2).
- Pooling turns RQ1's null into a bounded result: Jev's action policy is within about ±1
  score point of static, and even a perfect action policy would gain under 3 points.
  *[revised; see Revisions item 9: in a controlled test no operator beats the `echo` no-op, so
  operator choice is not where the leverage is either.]*
- The cluster analysis shows what the score metric hides: Jev's action policy measurably
  narrows attack diversity. Meanwhile, the attack family (corpus cluster) determines
  refusal and success more than any selector. Only 3 of 26 clusters ever produced a
  success.

## Recommended next decisions
(Items 1–4 cost API budget. With no budget, the pooled analyses above are the result, and the
framing is a bounded negative result plus a mechanism study.)
1. ~~Run the referee~~ *[done: refusal-gated Meta referee on Fit=Jev, plus a blind Claude
   annotation of 103 prompts; Revisions items 4 and 5.]*
2. **Increase power where it matters.** Use more seeds (≥15/arm) and more mutations per
   iteration, or a less refusal-prone target, so successes aren't a handful of lucky seeds.
   More iterations alone won't help, because the search already stalls by iteration 3.
3. **Fix the coverage objective's cold start.** For example, give partial credit for
   `low_risk`/`is_related`, or fall back to the score objective until the first category is
   covered.
4. *[withdrawn]* The per-task seeding is the system's configuration and stays unchanged. Instead,
   log intended and realized choices side by side, as this study did, and quantify the coupling
   offline.
5. *[done]* RQ3 investigated with the five-framing description ablation (Revisions item 2).
