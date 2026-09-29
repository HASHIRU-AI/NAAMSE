"""Plain zero-shot LLM baseline for the decision points: Muse Spark names one option.

Receives exactly what Jev receives (the same instructions, option descriptions, and parent
state), but as a chat prompt answered in free text at temperature 0, so it returns a single
choice rather than a probability per option. Because the choice does not depend on the random
number generator, NAAMSE's shared per-task seed cannot distort it.
"""
import json
import logging
import os
import random
import re
import threading
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_STATE_CHARS = 8000
_client = None
_client_lock = threading.Lock()


def get_muse_client():
    """Shared Muse Spark chat model (MUSE_SELECTOR_MODEL overrides the model id)."""
    global _client
    with _client_lock:
        if _client is None:
            from src.behavioral_engine.moe_score_subgraph.llm_judges.meta_judge import MetaJudge
            model = os.getenv("MUSE_SELECTOR_MODEL", "muse-spark-1.2")
            _client = MetaJudge(model_name=model, temperature=0).get_model()
        return _client


def _norm(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.lower())


def parse_choice(reply: str, options: Iterable[str]) -> Optional[str]:
    """The option named earliest in the reply (longest name wins ties); None if none is named."""
    text = _norm(reply)
    hits = [(text.find(_norm(o)), -len(o), o) for o in options if _norm(o) in text]
    return min(hits)[2] if hits else None


def build_prompt(instructions: str, criteria: Dict[str, str], state: Any) -> str:
    options = "\n".join(f"- {name}: {desc}" for name, desc in criteria.items())
    state_json = json.dumps(state, default=str)[:MAX_STATE_CHARS]
    return (f"{instructions}\n\nState (JSON):\n{state_json}\n\nOptions:\n{options}\n\n"
            "Reply with exactly one option name from the list above, and nothing else.")


def ask_muse_choice(question_id: str, instructions: str, criteria: Dict[str, str], state: Any,
                    rng: random.Random, client=None) -> Tuple[str, Dict[str, float]]:
    """Ask Muse Spark to pick one option. Returns (choice, {choice: 1.0}).

    If the reply names no option, falls back to a random option and returns an empty
    probability map, so parse failures can be counted from the run logs.
    """
    client = client or get_muse_client()
    reply = client.invoke(build_prompt(instructions, criteria, state)).content
    reply = reply if isinstance(reply, str) else str(reply)
    choice = parse_choice(reply, criteria)
    if choice is None:
        logger.warning("Muse %s reply named no option (%r); falling back to random", question_id, reply[:120])
        return rng.choice(sorted(criteria)), {}
    return choice, {choice: 1.0}
