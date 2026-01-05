from enum import Enum
from typing_extensions import List, Literal, TypedDict, Optional


class Mutation(Enum):
    ADVERSARIAL_PREFIX = "adversarial_prefix_mutation"
    PAYLOAD_SPLITTING = "payload_splitting"
    MATHEMATICAL_ATTACK = "mathematical_attack"
    UNICODE_MUTATION = "unicode_mutation"
    NARRATIVE_DISPLACEMENT = "narrative_displacement"
    DEEP_INCEPTION_MUTATION = "deep_inception_mutation"
    CODE_EXEC = "code_exec"
    EMOJI = "emoji"
    MEMORY_PREPEND = "memory_prepend"
    ECHO = "echo"
    CIPHER_MUTATION = "cipher_mutation"
    ARTPROMPT = "artprompt"
    MANY_SHOT_JAILBREAKING = "many_shot_jailbreaking"
    TASK_CONCURRENCY_ATTACK = "task_concurrency_attack"
    GAME_THEORY_ATTACK = "game_theory_attack"
    ADVERSARIAL_POETRY_MUTATION = "adversarial_poetry_mutation"
    PERSONA_ROLEPLAY_MUTATION = "persona_roleplay_mutation"
    DUAL_RESPONSE_DIVIDER_MUTATION = "dual_response_divider_mutation"
    CONTEXTUAL_FRAMING_MUTATION = "contextual_framing_mutation"
    DARKCITE = "darkcite"
    LANGUAGE_GAMES_MUTATION = "language_games_mutation"
    SATA_ASSISTIVE_TASK_MUTATION = "sata_assistive_task_mutation"
    SEMANTIC_STEGANOGRAPHY_MUTATION = "semantic_steganography_mutation"
    SYNONYM_MUTATION = "synonym_mutation"
    EXPLORE = "explore"
    SIMILAR = "similar"


class ClusterInfo(TypedDict, total=False):
    """cluster information from the cluster engine."""
    cluster_id: str
    cluster_label: str


class Metadata(TypedDict, total=False):
    cluster_info: ClusterInfo
    mutation_type: 'Mutation'

class BasePrompt(TypedDict, total=False):
    """Base prompt structure with metadata."""
    prompt: List[str]
    metadata: Metadata


from src.behavioral_engine.behavior_engine_workflow_state import ConversationHistory

class ScoredPrompt(BasePrompt):
    """Prompt with associated score."""
    score: float
    conversation_history: Optional[ConversationHistory] = None


class MutatedPrompt(BasePrompt):
    """Prompt with associated score."""
    mutation_type: Mutation


class MutationWorkflowState(TypedDict):
    """State for the mutation subgraph."""
    prompt_to_mutate: BasePrompt
    mutated_prompt: BasePrompt | MutatedPrompt
    mutation_type: Mutation
    task_seed: Optional[int]  # Seed for deterministic mutation selection


class MutationEngineState(TypedDict):
    """Complete state for the mutation engine workflow."""
    # Inputs
    input_prompts: List[ScoredPrompt]
    n_to_generate: int
    task_seed: Optional[int]  # Seed for this specific parallel task

    # Internal state
    prompt_probabilities: List[float]
    selected_prompt: Optional[ScoredPrompt]
    action_to_take: Literal["mutate", "similar", "explore"]
    newly_generated_prompt: Optional[BasePrompt]

    # Output
    final_generated_prompts: List[BasePrompt]
