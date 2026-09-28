"""Shared helpers for asking Jev a single choice question and sampling from its probabilities."""
import random
from typing import Any, Dict, List, Optional, Tuple

from src.behavioral_engine.moe_score_subgraph.llm_judges.jev_judge import JevJudge

MAX_PROMPT_CHARS = 8000  # keep Jev's `state` well inside its 32k-token state budget

_client = None


def get_jev_client():
    """Shared Jev client (model and base URL come from JEV_MODEL / TYPESAFE_API_BASE_URL)."""
    global _client
    if _client is None:
        _client = JevJudge().get_model()
    return _client


def sample_choice(probabilities: Dict[str, float], temperature: float, rng: random.Random) -> str:
    """Sample a key from Jev's probabilities; temperature 0 means argmax."""
    options = sorted(probabilities)
    if temperature == 0:
        return max(options, key=lambda k: probabilities[k])
    weights = [max(probabilities[k], 0.0) ** (1.0 / temperature) for k in options]
    if sum(weights) == 0:
        return rng.choice(options)
    return rng.choices(options, weights=weights, k=1)[0]


def ask_jev_choice(question_id: str, instructions: str, criteria: Dict[str, str], state: Any,
                   temperature: float, rng: random.Random, client=None) -> Tuple[str, Dict[str, float]]:
    """Ask Jev one choice question; return (sampled choice, probabilities)."""
    client = client or get_jev_client()
    answer = client.evaluate(state, {
        question_id: {"type": "choice", "instructions": instructions, "criteria": criteria},
    })[question_id]
    probabilities = answer.get("probabilities") or {answer["choice"]: 1.0}
    unknown = set(probabilities) - set(criteria)
    if unknown:
        raise ValueError(f"Jev returned options outside the criteria: {sorted(unknown)}")
    return sample_choice(probabilities, temperature, rng), probabilities


def describe_parent(prompt: Dict[str, Any], uncovered_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """Jev `state` describing a parent prompt and its judge feedback."""
    text = " ".join(str(p) for p in prompt.get("prompt", []))
    metadata = prompt.get("metadata", {}) or {}
    judge = metadata.get("judge_results", {}) or {}
    state: Dict[str, Any] = {
        "parent_prompt": text[:MAX_PROMPT_CHARS],
        "parent_score": round(float(prompt.get("score", 0.0)), 2),
        "parent_judge_feedback": {
            cat: {"is_related": r.get("is_related"), "harm": r.get("harm")}
            for cat, r in judge.get("categories", {}).items()
        },
        "target_response_alignment": judge.get("alignment"),
    }
    if uncovered_categories is not None:
        state["harm_categories_not_yet_covered"] = uncovered_categories
    return state
