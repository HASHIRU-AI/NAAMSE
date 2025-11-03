def run_similar_action(state):
    """3b. Fetches a similar prompt."""
    print("--- [Mutation Engine] Running Action: SIMILAR ---")
    return {"newly_generated_prompt": f"similar_to_({state['selected_prompt'][0]})"}