from typing import Dict, Any
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from src.behavioral_engine.moe_score_subgraph.more_score_state import Context, MOEResponse, JudgeResult, MOESubgraphState
from src.behavioral_engine.moe_score_subgraph.llm_judges.llm_judge import LLMJudge


def create_judge_node(judge: LLMJudge):
    """
    Factory function to create a judge node for a specific LLM judge
    
    Returns a node function that can be added to the subgraph
    """
    
    judge_prompt = judge.get_system_prompt()

    # Create the agent for this judge (done once at graph construction time)
    eval_agent = create_agent(
        model=judge.get_model(),
        system_prompt=judge_prompt,
        response_format=ToolStrategy(MOEResponse)
    )
    
    def judge_node(state: MOESubgraphState) -> Dict[str, Any]:
        """
        Node function that evaluates conversation using this judge
        Runs in parallel with other judges
        """
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
        
        result = JudgeResult(
            judge_name=judge.get_name(),
            eval_type=judge.get_eval_type(),
            is_related=response.is_related,
            is_prompt_harmful=response.is_prompt_harmful,
            is_refusal=response.is_refusal,
            is_response_harmful=response.is_response_harmful
        )
        
        print(f"[{judge.get_name()}] Result: is_related={response.is_related}, is_prompt_harmful={response.is_prompt_harmful}, is_refusal={response.is_refusal}, is_response_harmful={response.is_response_harmful}")
        
        # This gets accumulated in state due to operator.add annotation
        return {
            "judge_results": [result]
        }
    
    return judge_node
