from typing_extensions import List, Literal, TypedDict, Optional, Dict

class ConversationHistory(TypedDict):
    """Structure for conversation history."""
    messages: List[Dict[str, str]]
    
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
    moe_score: int

    #outputs
    final_score: float