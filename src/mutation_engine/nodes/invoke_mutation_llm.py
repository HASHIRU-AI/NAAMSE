from src.mutation_engine.mutation_workflow_state import BasePrompt, MutationWorkflowState


def invoke_llm_with_tools(state: MutationWorkflowState):
    """2. Invokes LLM (placeholder) to perform the mutation."""
    print(
        f"  [Mutation Subgraph] Invoking LLM to mutate: '{state['prompt_to_mutate']}'")
    mutated_prompt = BasePrompt(
        prompt=[f"Mutated version of: {state['prompt_to_mutate']['prompt'][0]}"]
    )
    return {"mutated_prompt": mutated_prompt}
