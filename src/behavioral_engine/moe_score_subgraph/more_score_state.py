from typing import Literal, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from src .behavioral_engine.behavior_engine_workflow_state import ConversationHistory
class Context(TypedDict):
    eval_type: str

class MOEResponse(BaseModel):
    """Response from LLM judge evaluation"""
    result: bool

class JudgeResult(BaseModel):
    """Result from a single judge evaluation"""
    score: int = Field(description="Score from 0-100")
    judge_name: str = Field(description="Name of the judge")

# Subgraph State
class MOESubgraphState(TypedDict):
    """State for MOE evaluation subgraph"""
    conversation_history: ConversationHistory
    judge_results: Annotated[List[JudgeResult], operator.add]  # Accumulate results from parallel judges
    aggregation_method: str
    moe_score: float