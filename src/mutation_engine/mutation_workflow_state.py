from enum import Enum
from typing_extensions import List, Literal, TypedDict, Optional


class Mutation(Enum):
    SUBSTITUTION = "substitution"
    ADVERSARIAL_PREFIX = "adversarial_prefix_mutation"
    PAYLOAD_SPLITTING = "payload_splitting"
    MATHEMATICAL_ATTACK = "mathematical_attack"
    UNICODE_MUTATION = "unicode_mutation"
    NARRATIVE_DISPLACEMENT = "narrative_displacement"
    DEEP_INCEPTION_MUTATION = "deep_inception_mutation"
    CODE_EXEC = "code_exec"
    EMOJI = "emoji"
    MEMORY_PREPEND = "memory_prepend"


class BasePrompt(TypedDict):
    """Base prompt structure."""
    prompt: List[str]


class ScoredPrompt(BasePrompt):
    """Prompt with associated score."""
    score: float


class MutatedPrompt(BasePrompt):
    """Prompt with associated score."""
    mutation_type: Mutation


class MutationWorkflowState(TypedDict):
    """State for the mutation subgraph."""
    prompt_to_mutate: BasePrompt
    mutated_prompt: BasePrompt | MutatedPrompt
    mutation_type: Mutation


class MutationEngineState(TypedDict):
    """Complete state for the mutation engine workflow."""
    # Inputs
    input_prompts: List[ScoredPrompt]
    n_to_generate: int

    # Internal state
    prompt_probabilities: List[float]
    selected_prompt: Optional[ScoredPrompt]
    action_to_take: Literal["mutate", "similar", "explore"]
    newly_generated_prompt: Optional[BasePrompt]

    # Output
    final_generated_prompts: List[BasePrompt]
