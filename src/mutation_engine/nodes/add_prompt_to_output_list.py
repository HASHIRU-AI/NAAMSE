def add_prompt_to_output_list(state):
    """Adds the newly created prompt to the final list."""
    new_prompt = state["newly_generated_prompt"]
    current_list = state["final_generated_prompts"]
    current_list.append(new_prompt)
    print(
        f"--- [Mutation Engine] Added to output. Total generated: {len(current_list)} ---")
    return {"final_generated_prompts": current_list}