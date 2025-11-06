from src.mutation_engine.mutation_workflow_state import BasePrompt, MutatedPrompt, MutationWorkflowState
from langchain_google_genai import ChatGoogleGenerativeAI

def invoke_llm(prompt: str) -> BasePrompt:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm = llm.with_structured_output(BasePrompt, method="json_schema")
    return llm.invoke(prompt)


def invoke_llm_with_tools(state: MutationWorkflowState):
    """2. Invokes LLM (placeholder) to perform the mutation."""
    print(
        f"  [Mutation Subgraph] Invoking LLM to mutate: '{state['prompt_to_mutate']}'")
    
    mutation_instruction = {
        "paraphrase": "Paraphrase the following prompt:",
        "add_negation": "Add negation to the following prompt:",
        "synonym_swap": "Swap some words in the following prompt with their synonyms:",
        "split_prompt": "Split the following prompt into two separate prompts:"
    }[state['mutation_type']]
    llm_prompt = f"{mutation_instruction}\n\nPrompt: {state['prompt_to_mutate']['prompt']}"
    output: BasePrompt = invoke_llm(llm_prompt)
    
    return {"mutated_prompt": MutatedPrompt(prompt=output['prompt'], mutation_type=state['mutation_type'])}
