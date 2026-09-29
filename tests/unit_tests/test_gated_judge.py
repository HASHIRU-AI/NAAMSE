"""Tests for the refusal-gated scoring graph used by the referee.

Judges are replaced by stubs, so no API is called. The key property: whenever the harm judges
would have answered "not related / not harmful" on a refusal (true for all 754 refusals in the
Jev sweep), the gated graph returns the same scores as the full graph while skipping the six
harm-judge calls.
"""
from typing import Dict, List

import pytest

from src.behavioral_engine.moe_score_subgraph import moe_score_workflow
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType
from src.behavioral_engine.moe_score_subgraph.more_score_state import (
    HarmAssessmentLevel, JudgeResult, RAJudgeResult, ResponseAlignment,
)
from src.experiments import gated_judge
from src.experiments.gated_judge import GATED_JUDGE_NAME, HARM_NODES, HARM_TYPES, gated_harm, judge_calls

STATE = {
    "sanitized_conversation_history": {"messages": [{"role": "user", "content": "q"},
                                                    {"role": "assistant", "content": "a"}]},
    "judge_results": [],
}


@pytest.fixture
def stub_judges(monkeypatch):
    """Replace every judge with a stub; returns a setter for the alignment and a call log."""
    calls: List[str] = []
    setting: Dict[str, object] = {"alignment": ResponseAlignment.FULL_REFUSAL,
                                  "harm": HarmAssessmentLevel.NOT_HARMFUL, "related": False}

    def fake_get_judge_node(backend: str, eval_type: EvalType):
        def node(state):
            calls.append(eval_type.value)
            if eval_type == EvalType.RESPONSE_ALIGNMENT:
                return {"ra_judge_results": RAJudgeResult(judge_name="stub", eval_type=eval_type,
                                                          is_response_aligned=setting["alignment"])}
            return {"judge_results": [JudgeResult(judge_name="stub", eval_type=eval_type,
                                                  is_related=setting["related"],
                                                  is_response_harmful=setting["harm"])]}
        return node

    monkeypatch.setattr(moe_score_workflow, "get_judge_node", fake_get_judge_node)
    return setting, calls


CONFIG = {"configurable": {"judge_backend": "meta"}}


@pytest.mark.parametrize("alignment", [ResponseAlignment.FULL_REFUSAL, ResponseAlignment.WEAK_REFUSAL])
def test_refusal_skips_harm_judges(stub_judges, alignment):
    setting, calls = stub_judges
    setting["alignment"] = alignment
    out = gated_judge._build_gated_moe_graph().invoke(dict(STATE), config=CONFIG)
    assert calls == [EvalType.RESPONSE_ALIGNMENT.value]
    assert len(out["judge_results"]) == len(HARM_TYPES)
    assert all(r.judge_name == GATED_JUDGE_NAME for r in out["judge_results"])
    assert judge_calls(out["judge_results"]) == 1


@pytest.mark.parametrize("alignment", [ResponseAlignment.WEAK_ACCEPTANCE, ResponseAlignment.FULL_ACCEPTANCE])
def test_acceptance_calls_all_harm_judges(stub_judges, alignment):
    setting, calls = stub_judges
    setting.update(alignment=alignment, related=True, harm=HarmAssessmentLevel.LOW_RISK)
    out = gated_judge._build_gated_moe_graph().invoke(dict(STATE), config=CONFIG)
    assert sorted(calls) == sorted([EvalType.RESPONSE_ALIGNMENT.value] + [t.value for t in HARM_TYPES])
    assert judge_calls(out["judge_results"]) == 1 + len(HARM_TYPES)
    assert out["harm_score"] == 2  # low_risk


@pytest.mark.parametrize("alignment", list(ResponseAlignment))
def test_gated_scores_match_full_graph_when_refusals_are_harmless(stub_judges, alignment):
    setting, _ = stub_judges
    setting.update(alignment=alignment, related=False, harm=HarmAssessmentLevel.NOT_HARMFUL)
    full = moe_score_workflow.moe_score_graph.invoke(dict(STATE), config=CONFIG)
    gated = gated_judge._build_gated_moe_graph().invoke(dict(STATE), config=CONFIG)
    assert (gated["ra_score"], gated["harm_score"]) == (full["ra_score"], full["harm_score"])


def test_gated_harm_is_not_related_and_not_harmful():
    results = gated_harm(STATE)["judge_results"]
    assert [r.eval_type for r in results] == HARM_TYPES
    assert all(not r.is_related and r.is_response_harmful == HarmAssessmentLevel.NOT_HARMFUL for r in results)
    assert len(HARM_NODES) == len(HARM_TYPES) == 6
