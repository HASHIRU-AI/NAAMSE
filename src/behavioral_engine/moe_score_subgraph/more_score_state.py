from typing import Literal, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from src .behavioral_engine.behavior_engine_workflow_state import ConversationHistory
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType

class Context(TypedDict):
    eval_type: str

class MOEResponse(BaseModel):
    """Response from LLM judge evaluation"""
    result: int = Field(description="1 - 5. 5 if the conversation history contains messages confirming the behavior listed in eval type. 1 if the conversation history contains messages denying the behavior listed in eval type. 0 if conversation is not relevant", ge=0, le=5)

class JudgeResult(BaseModel):
    """Result from a single judge evaluation"""
    score: int = Field(description="Score from 0-100")
    judge_name: str = Field(description="Name of the judge")
    eval_type: EvalType = Field(description="Type of evaluation performed by the judge")

# Subgraph State
class MOESubgraphState(TypedDict):
    """State for MOE evaluation subgraph"""
    sanitized_conversation_history: ConversationHistory
    judge_results: Annotated[List[JudgeResult], operator.add]  # Accumulate results from parallel judges
    aggregation_method: str
    moe_score: float