from src.mutation_engine.mutation_workflow_state import BasePrompt, MutationEngineState

def add_prompt_to_output_list(state: MutationEngineState) -> dict:
    """Adds the newly created prompt to the final list, with the selector decisions that produced it."""
    new_prompt: BasePrompt = dict(state["newly_generated_prompt"])
    metadata = dict(new_prompt.get("metadata") or {})
    selector = dict(metadata.get("selector") or {})
    selector["action"] = state.get("action_to_take")
    selector["action_probabilities"] = state.get("action_probabilities")
    metadata["selector"] = selector
    new_prompt["metadata"] = metadata

    current_list = state["final_generated_prompts"]
    current_list.append(new_prompt)
    print(
        f"--- [Mutation Engine] Added to output. Total generated: {len(current_list)} ---")
    return {"final_generated_prompts": current_list}
