from langgraph.graph import END

from src.mutation_engine.mutation_workflow_state import MutationEngineState


def should_continue_loop(state: MutationEngineState) -> MutationEngineState:
    """Conditional check: Have we generated 'n' prompts yet?"""
    if len(state["final_generated_prompts"]) < state["n_to_generate"]:
        return "select_prompt"  # Loop back
    else:
        return END  # Exit loop