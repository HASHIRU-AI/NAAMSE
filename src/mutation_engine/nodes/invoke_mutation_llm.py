from src.mutation_engine.mutation_workflow_state import BasePrompt, MutatedPrompt, MutationWorkflowState
from langchain_google_genai import ChatGoogleGenerativeAI


def invoke_llm_with_tools(state: MutationWorkflowState):
    """2. Invokes LLM (placeholder) to perform the mutation."""
    print(
        f"  [Mutation Subgraph] Invoking LLM to mutate: '{state['prompt_to_mutate']}'")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm = llm.with_structured_output(BasePrompt, method="json_schema")
    output = llm.invoke(f'Mutate the following prompt using the mutation type "{state["mutation_type"]}": {state["prompt_to_mutate"]["prompt"]}')
    print(f"  [Mutation Subgraph] LLM Output: {output}")
    
    return {"mutated_prompt": MutatedPrompt(prompt=output['prompt'], mutation_type=state['mutation_type'])}
