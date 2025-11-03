from typing_extensions import TypedDict

class MutationWorkflowState(TypedDict):
    """State for the mutation subgraph."""
    prompt_to_mutate: str
    mutated_prompt: str
