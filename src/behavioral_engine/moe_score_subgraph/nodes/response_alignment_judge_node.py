from typing import Dict, Any
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from src.behavioral_engine.moe_score_subgraph.more_score_state import Context, RAJudgeResponse, RAJudgeResult, MOESubgraphState, ResponseAlignment
from src.behavioral_engine.moe_score_subgraph.llm_judges.llm_judge import LLMJudge
import os
from dotenv import load_dotenv

def create_response_alignment_judge_node(judge: LLMJudge):

    judge_prompt = judge.get_system_prompt()

    # Create the agent for this judge (done once at graph construction time)
    eval_agent = create_agent(
        model=judge.get_model(),
        system_prompt=judge_prompt,
        response_format=ToolStrategy(RAJudgeResponse)
    )
    
    def judge_node(state: MOESubgraphState) -> Dict[str, Any]:
        """
        Node function that evaluates conversation using this judge
        Runs in parallel with other judges
        """
        load_dotenv()  # Load environment variables from .env file
        skip_llm = os.getenv("SKIP_LLM", "false").lower() == "true"
        if skip_llm:
            print(f"[{judge.get_name()}] Skipping LLM invocation as per configuration.")
            # Return a default JudgeResult indicating no issues
            result = RAJudgeResult(
                judge_name=judge.get_name(),
                eval_type=judge.get_eval_type(),
                is_response_aligned=ResponseAlignment.FULL_ACCEPTANCE,
            )
            return {
                "ra_judge_results": result
            }
        conversation_history = state["sanitized_conversation_history"]
        
        print(f"\n[{judge.get_name()}] Starting evaluation...")
            
        response = eval_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"""You have to analyze the following interaction:
                            Input Prompt: {conversation_history["messages"][0]["content"]}
                            AI Response: {conversation_history["messages"][1:]}"""
            }],
        })["structured_response"]
        
        result = RAJudgeResult(
            judge_name=judge.get_name(),
            eval_type=judge.get_eval_type(),
            is_response_aligned=response.is_response_aligned
        )
        
        print(f"[{judge.get_name()}] Result: is_response_aligned={response.is_response_aligned}")
        
        # This gets accumulated in state due to operator.add annotation
        return {
            "ra_judge_results": result
        }
    
    return judge_node