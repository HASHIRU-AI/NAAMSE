"""Tests for the ablation config, coverage/fitness, and the action/mutation selectors.

The key property: default flags reproduce the original NAAMSE selection exactly,
draw for draw, so the baseline arm is the unmodified system.
"""
import random
from types import SimpleNamespace

import pytest

from src.experiments.ablation_config import AblationConfig, add_ablation_args, ablation_from_args, get_ablation
from src.experiments.coverage import (
    HARM_CATEGORIES, compute_fitness, coverage_fitness, covered_categories, harmful_categories,
)
from src.mutation_engine.mutation_workflow_state import Mutation
from src.mutation_engine.selectors.action_selectors import JEV_ACTION_CRITERIA, select_action
from src.mutation_engine.selectors.jev_decisions import ask_jev_choice, sample_choice
from src.mutation_engine.selectors.mutation_descriptions import MUTATION_DESCRIPTIONS, MUTATION_OPERATORS
from src.mutation_engine.selectors.mutation_selectors import select_mutation


def _original_action(score: float, rng: random.Random) -> str:
    """The pre-change decide_action_by_score logic, copied verbatim for comparison."""
    if score < 50:
        weights = [0.7, 0.2, 0.1]
    elif score < 80:
        weights = [0.1, 0.7, 0.2]
    elif score < 100:
        weights = [0.1, 0.2, 0.7]
    else:
        weights = [0.4, 0.4, 0.2]
    return rng.choices(["explore", "similar", "mutate"], weights=weights, k=1)[0]


def _original_mutation(rng: random.Random) -> str:
    mutations = list(Mutation)
    mutations.remove(Mutation.SIMILAR)
    mutations.remove(Mutation.EXPLORE)
    return rng.choice(mutations).value


class FakeJev:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        return {qid: self.answer for qid in questions}


# --- ablation config ---

def test_defaults_are_the_baseline_arm():
    a = AblationConfig()
    assert (a.action_selector, a.mutation_selector, a.fitness_judge, a.objective) == (
        "static_thresholds", "uniform", "gemini", "score")


@pytest.mark.parametrize("field,value", [
    ("action_selector", "nope"), ("mutation_selector", "nope"), ("fitness_judge", "nope"),
    ("referee_judge", "nope"), ("objective", "nope"), ("coverage_weight", 1.5), ("jev_temperature", -1.0),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValueError):
        AblationConfig(**{field: value})


def test_cli_flags_round_trip():
    import argparse
    parser = argparse.ArgumentParser()
    add_ablation_args(parser)
    args = parser.parse_args(["--action-selector", "jev", "--mutation-selector", "jev", "--fitness-judge", "meta",
                              "--objective", "combined", "--coverage-weight", "0.3", "--jev-temperature", "0"])
    a = ablation_from_args(args)
    assert (a.action_selector, a.mutation_selector, a.fitness_judge) == ("jev", "jev", "meta")
    assert (a.objective, a.coverage_weight, a.jev_temperature) == ("combined", 0.3, 0.0)


def test_get_ablation_defaults_and_reads_config():
    assert get_ablation(None) == AblationConfig()
    assert get_ablation({"configurable": {}}) == AblationConfig()
    arm = AblationConfig(action_selector="uniform")
    assert get_ablation({"configurable": {"ablation": arm}}) is arm


# --- baseline equivalence ---

@pytest.mark.parametrize("score", [0, 10, 49.9, 50, 79.9, 80, 99.9, 100])
def test_static_thresholds_matches_original_draw_for_draw(score):
    for seed in range(50):
        got, probs = select_action({"score": score}, random.Random(seed), AblationConfig())
        assert got == _original_action(score, random.Random(seed))
        assert probs is None


def test_uniform_mutation_matches_original_draw_for_draw():
    for seed in range(100):
        got, probs = select_mutation({"prompt": ["x"]}, random.Random(seed), AblationConfig())
        assert got == _original_mutation(random.Random(seed))
        assert probs is None


# --- descriptions ---

def test_every_operator_has_a_description():
    assert len(MUTATION_OPERATORS) == 26
    assert set(MUTATION_OPERATORS) == set(MUTATION_DESCRIPTIONS)
    assert all(d.strip().endswith(".") for d in MUTATION_DESCRIPTIONS.values())


# --- Jev selectors ---

def test_sample_choice_argmax_at_zero_temperature():
    assert sample_choice({"a": 0.2, "b": 0.7, "c": 0.1}, 0, random.Random(0)) == "b"


def test_sample_choice_follows_probabilities():
    rng = random.Random(0)
    draws = [sample_choice({"a": 0.9, "b": 0.1}, 1.0, rng) for _ in range(2000)]
    assert 0.85 < draws.count("a") / len(draws) < 0.95


def test_jev_action_selector_uses_returned_probabilities():
    jev = FakeJev({"type": "choice", "choice": "mutate",
                   "probabilities": {"explore": 0.05, "similar": 0.05, "mutate": 0.9}})
    ablation = AblationConfig(action_selector="jev", jev_temperature=0)
    prompt = {"prompt": ["p"], "score": 88.0}
    from src.mutation_engine.selectors import jev_decisions
    jev_decisions._client = jev
    try:
        action, probs = select_action(prompt, random.Random(0), ablation)
    finally:
        jev_decisions._client = None
    assert action == "mutate"
    assert probs["mutate"] == 0.9
    state, questions = jev.calls[0]
    assert state["parent_score"] == 88.0
    assert set(questions["next_action"]["criteria"]) == set(JEV_ACTION_CRITERIA)


def test_jev_mutation_selector_offers_all_operators():
    choice = MUTATION_OPERATORS[3].value
    jev = FakeJev({"type": "choice", "choice": choice, "probabilities": {choice: 1.0}})
    from src.mutation_engine.selectors import jev_decisions
    jev_decisions._client = jev
    try:
        picked, _ = select_mutation({"prompt": ["p"], "score": 10.0}, random.Random(0),
                                    AblationConfig(mutation_selector="jev", jev_temperature=0))
    finally:
        jev_decisions._client = None
    assert picked == choice
    criteria = jev.calls[0][1]["mutation"]["criteria"]
    assert len(criteria) == 26 and criteria[choice] == MUTATION_DESCRIPTIONS[MUTATION_OPERATORS[3]]


def test_jev_option_outside_criteria_is_an_error():
    jev = FakeJev({"type": "choice", "choice": "bogus", "probabilities": {"bogus": 1.0}})
    with pytest.raises(ValueError):
        ask_jev_choice("q", "instructions", {"a": "A", "b": "B"}, {}, 1.0, random.Random(0), client=jev)


# --- coverage / fitness ---

def _summary(**harm_by_cat):
    return {"categories": {c: {"is_related": True, "harm": h} for c, h in harm_by_cat.items()}}


def test_harmful_categories_needs_related_and_high_risk():
    s = {"categories": {
        "violence": {"is_related": True, "harm": "harmful"},
        "non_violent_crime": {"is_related": True, "harm": "low_risk"},
        "disinformation_deception": {"is_related": False, "harm": "harmful"},
    }}
    assert harmful_categories(s) == ["violence"]
    assert harmful_categories(None) == []


def test_covered_categories_is_a_union():
    a, b = _summary(violence="harmful"), _summary(non_violent_crime="high_risk")
    assert covered_categories([a, b, None]) == ["non_violent_crime", "violence"]


def test_coverage_fitness_ignores_already_covered_categories():
    s = _summary(violence="harmful", non_violent_crime="high_risk")
    assert coverage_fitness(s, []) == 100.0
    assert coverage_fitness(s, ["violence"]) == pytest.approx(200 / 3)
    assert coverage_fitness(s, list(HARM_CATEGORIES)) == 0.0


@pytest.mark.parametrize("objective,expected", [("score", 40.0), ("coverage", 100.0), ("combined", 70.0)])
def test_compute_fitness_by_objective(objective, expected):
    a = AblationConfig(objective=objective, coverage_weight=0.5)
    assert compute_fitness(40.0, _summary(violence="harmful"), [], a) == pytest.approx(expected)


def test_default_objective_leaves_score_untouched():
    assert compute_fitness(63.5, None, [], AblationConfig()) == 63.5
