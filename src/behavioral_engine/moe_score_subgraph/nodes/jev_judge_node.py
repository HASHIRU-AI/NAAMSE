from typing import Dict, Any
from src.behavioral_engine.moe_score_subgraph.more_score_state import JudgeResult, RAJudgeResult, MOESubgraphState, HarmAssessmentLevel, ResponseAlignment
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType
from src.behavioral_engine.moe_score_subgraph.llm_judges.jev_judge import JevJudge
import os
from dotenv import load_dotenv

def create_jev_judge_node(judge: JevJudge):
    """
    Factory function to create a node for a Jev judge.

    Jev answers typed questions instead of chatting, so it cannot go through
    create_agent. Returns the same state updates as create_judge_node
    (harm categories) or create_response_alignment_judge_node (response alignment).
    """

    client = judge.get_model()
    questions = judge.build_questions()
    is_ra_judge = judge.get_eval_type() == EvalType.RESPONSE_ALIGNMENT

    def judge_node(state: MOESubgraphState) -> Dict[str, Any]:
        load_dotenv()  # Load environment variables from .env file
        skip_llm = os.getenv("SKIP_LLM", "false").lower() == "true"
        if skip_llm:
            print(f"[{judge.get_name()}] Skipping LLM invocation as per configuration.")
            if is_ra_judge:
                return {"ra_judge_results": RAJudgeResult(
                    judge_name=judge.get_name(),
                    eval_type=judge.get_eval_type(),
                    is_response_aligned=ResponseAlignment.FULL_ACCEPTANCE,
                )}
            return {"judge_results": [JudgeResult(
                judge_name=judge.get_name(),
                eval_type=judge.get_eval_type(),
                is_related=False,
                is_response_harmful=HarmAssessmentLevel.NOT_HARMFUL,
            )]}

        messages = state["sanitized_conversation_history"]["messages"]
        jev_state = {
            "input_prompt": messages[0]["content"],
            "conversation_history": messages[1:],
        }

        print(f"\n[{judge.get_name()}] Starting evaluation...")
        if client.api_key is None:
            # Pick up a key loaded by load_dotenv() after graph construction
            client.api_key = os.getenv("TYPESAFE_API_KEY")
        answers = client.evaluate(jev_state, questions)

        if is_ra_judge:
            result = RAJudgeResult(
                judge_name=judge.get_name(),
                eval_type=judge.get_eval_type(),
                is_response_aligned=judge.parse_alignment_answers(answers),
                alignment_probabilities=answers["response_alignment"].get("probabilities"),
            )
            print(f"[{judge.get_name()}] Result: is_response_aligned={result.is_response_aligned}")
            return {"ra_judge_results": result}

        is_related, is_response_harmful = judge.parse_harm_answers(answers)
        result = JudgeResult(
            judge_name=judge.get_name(),
            eval_type=judge.get_eval_type(),
            is_related=is_related,
            is_response_harmful=is_response_harmful,
            related_probability=answers["is_related"]["noul"],
            harm_probabilities=answers["is_response_harmful"].get("probabilities"),
        )
        print(f"[{judge.get_name()}] Result: is_related={is_related}, is_response_harmful={is_response_harmful}")
        return {"judge_results": [result]}

    return judge_node
