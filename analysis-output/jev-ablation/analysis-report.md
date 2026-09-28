# Jev Ablation Sweep: Analysis Report

Date: 2026-09-27 · Source: `outputs/` (40 runs) · Script: `scripts/analyze_ablation.py`

## Question
Does replacing NAAMSE's hand-built decision points with Jev probabilities change fuzzing
outcomes against the Muse Spark target?
- **RQ1**: Jev action selection (EXPLORE/SIMILAR/MUTATE) vs. static thresholds and uniform.
- **RQ2**: Jev as the fitness judge.
- **RQ3**: Jev mutation-operator choice vs. uniform.
- Each is run under the **score** and **coverage** objectives.

**Design.** 8 arms × 5 seeds, 7 iterations × 4 mutations = 28 prompts per run, 1,120 prompts
total. Mutation LLM: Meta. The fitness judge is Meta for every arm except Fit=Jev. **No referee
was run.**
**Unit of analysis.** One run. Runs are unpaired across arms (see caveat 2).

## Key findings

1. **No contrast shows a detectable effect on outcome metrics.** This covers mean score, max
   score, number of successes and coverage. Every Holm-adjusted p = 1.00, except max score for
   Cov+Mut=Jev (0.50). The 95% CIs are wide and include zero for every mean-score
   contrast; the largest is Act=Jev − Baseline = +2.7 [−4.2, +9.3]. With 5 runs per arm, the
   sweep can detect only very large effects. This is a null result under low power, **not
   evidence that the arms are equivalent**.

2. **Successes are too rare to compare arms.** Only 13 of 1,120 prompts (1.2%) reach score ≥ 80,
   the level that requires the judge to find harm. They come from 6 of the 40 runs, and 10 of
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

4. **RQ3: Jev's operator choice collapses to about 2 of 26 operators, with no gain.** In both
   objectives, 33 of Jev's 35 MUTATE picks were `semantic_steganography_mutation` (19) or
   `synonym_mutation` (14). Uniform selection used 12 different operators across 38 picks.
   Jev's MUTATE prompts score no better (mean 28.0 vs. 28.7; prompt-level Mann–Whitney
   p = 0.88), and neither selector produced any success from a MUTATE step. Jev's choices are
   nearly the same whatever the parent prompt, which fits a zero-shot preference based on the
   operator descriptions rather than on the parent.

5. **The coverage objective is degenerate at this success rate.** Coverage fitness is 0 until
   some category reaches high_risk, and that almost never happens. So every parent scores 0,
   and the static policy picks EXPLORE 70% of the time (observed 0.71). Jev also shifts to
   EXPLORE (P = 0.60; all 140 parents scored below 30). In practice the coverage arms are
   random search, so the "under coverage" contrasts don't test RQ1 or RQ3.

6. **RQ2 can't be evaluated from these runs.** The Fit=Jev arm is scored by Jev, while every
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

8. **Operators (RQ3): Jev's favorite operator is among the worst performers.**
   `semantic_steganography_mutation` accounts for 19 of Jev's 35 picks. Across 30 prompts in
   17 runs, it scores 23.3 [20.3, 26.5], with 90% refused. That's 19th of 25 operators and
   below every other operator with n ≥ 5 except two (deep_inception and
   adversarial_prefix). Operators Jev never picked do better:
   - artprompt: 41.5 [36.8, 50.5], n = 6, 1 success
   - dual_response_divider: 36.9, n = 7
   - many_shot_jailbreaking: 58.5, n = 2, 1 success

   Jev's other favorite, `synonym_mutation`, is mid-pack (34.9, n = 22). Operator outcomes
   vary widely (refusal from 0% to 100%), so operator choice *does* matter. The problem is
   that zero-shot selection from descriptions picked badly.

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
   - Without Meta labels on the same prompts, none of this says whether Jev is *accurate*.

## Main caveats
1. **Low power.** n = 5 runs per arm. The exact permutation test's minimum two-sided p is
   2/252 ≈ 0.008, before Holm correction over 7 contrasts.
2. **Seeds don't pair runs.** Same-seed runs in different arms start from different
   iteration-0 prompts (Act=Jev vs. Baseline share 3 of 4; Act=uniform shares 1 of 4), because
   each selector consumes the RNG differently. All tests are unpaired.
3. **No referee**, so all scores come from the fitness judge (Meta, or Jev for Fit=Jev). Muse
   Spark is both the target and the Meta judge, which risks self-preference bias.
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

## What changed in our understanding
- The current setup has too few successes to answer RQ1 or RQ3 on outcome metrics. The
  limiting factor is the target's refusal rate, not the selector.
- Jev's action policy (RQ1) is interpretable and behaves differently from the static policy.
  That is a result in itself (what Jev chooses), even though it didn't change outcomes.
- Jev's mutation selection (RQ3), as currently prompted, collapses to a narrow preference,
  and its favorite operator is one of the worst. It needs parent-conditioned context, or a
  check that the descriptions aren't biasing it, before a larger sweep is worth running.
- Pooling turns RQ1's null into a bounded result: Jev's action policy is within about ±1
  score point of static, and even a perfect action policy would gain under 3 points. RQ3 is
  where the leverage is: operators differ by more than 20 points.

## Recommended next decisions
(Items 1–4 cost API budget. With no budget, the pooled analyses above are the result, and the
framing is a bounded negative result plus a mechanism study.)
1. **Run the referee** (Meta, and Gemini as a cross-check) over existing `prompts.jsonl`
   conversations. This unblocks RQ2 and puts all arms on one scale, with no new fuzzing.
2. **Increase power where it matters.** Use more seeds (≥15/arm) and more mutations per
   iteration, or a less refusal-prone target, so successes aren't a handful of lucky seeds.
   More iterations alone won't help, because the search already stalls by iteration 3.
3. **Fix the coverage objective's cold start.** For example, give partial credit for
   `low_risk`/`is_related`, or fall back to the score objective until the first category is
   covered.
4. **Make seeds pair runs.** Use separate RNG streams for corpus sampling and for selector
   decisions, so same-seed arms share starting prompts and paired tests become valid.
5. **Investigate RQ3's collapse.** Check the probability entropy per call and the wording of
   `MUTATION_DESCRIPTIONS`.
