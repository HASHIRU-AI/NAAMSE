from typing_extensions import TypedDict

from src.behavioral_engine.behavior_engine_workflow_state import ConversationHistory
from src.mutation_engine.mutation_workflow_state import BasePrompt


class InvokeAgentWorkflowState(TypedDict):
    """
    This is the "interface" for the invoke agent workflow.
    It defines the inputs and outputs the parent graph will interact with.
    """
    # Inputs
    conversation_history: ConversationHistory

    a2a_agent_url: str

    prompt: BasePrompt
