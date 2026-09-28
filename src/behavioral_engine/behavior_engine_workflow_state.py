from typing_extensions import List, Literal, TypedDict, Optional, Dict, Any


class ConversationHistory(TypedDict):
    """Structure for conversation history."""
    messages: List[Dict[str, Any]]


class BehaviorEngineWorkflowState(TypedDict):
    """
    This is the "interface" for the main mutation engine.
    It defines the inputs and outputs the parent graph will interact with.
    """
    # Inputs
    conversation_history: ConversationHistory

    # Internal state
    decoded_conversation_history: ConversationHistory
    translated_conversation_history: ConversationHistory
    sanitized_conversation_history: ConversationHistory
    pii_score: int
    ra_score: int
    harm_score: int

    # outputs
    final_score: float
    # Per-category judge outputs from the MOE subgraph (JudgeResult / RAJudgeResult;
    # typed Any to avoid a circular import). Needed for category coverage.
    judge_results: List[Any]
    ra_judge_results: Any
    is_score_flipped: Optional[bool] = False  # Whether lower scores are worse
