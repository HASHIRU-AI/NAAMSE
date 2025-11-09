from typing_extensions import TypedDict
from src.mutation_engine.mutation_workflow_state import BasePrompt, MutatedPrompt, Mutation, MutationWorkflowState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    mutation_type: Mutation


@dynamic_prompt
def mutation_type_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on mutation type."""
    # get first enum value from Mutation enum
    defalt_mutation = list(Mutation)[0].value
    mutation_type = request.runtime.context.get(
        "mutation_type", defalt_mutation)
    base_prompt = "Given the following prompt, apply the mutation type: " + mutation_type + "."

    # fetch the user prompt from src.mutation_engine.nodes.mutations.{mutation_type}.system_prompt
    try:
        module_name = mutation_type
        module = __import__(
            f"src.mutation_engine.nodes.mutations.{module_name}", fromlist=[''])
        base_prompt = module.system_prompt
    except Exception as e:
        print(
            f"  [Mutation Subgraph] Could not load system prompt for mutation type: {mutation_type}. Using default prompt.")
        print(e)

    print(
        f"  [Mutation Subgraph] Using system prompt for mutation type '{mutation_type}': {base_prompt}")

    return base_prompt


def invoke_llm(prompt: BasePrompt, mutation: Mutation) -> BasePrompt:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # model = model.with_structured_output(BasePrompt, method="json_schema")
    # # go through each mutation in the enum and load corresponding tool from src/mutation_engine/nodes/mutations/
    tools = []
    # for mutation in Mutation:
    module_name = mutation.value
    module = __import__(
        f"src.mutation_engine.nodes.mutations.{module_name}", fromlist=[''])
    tool_function = getattr(module, module_name)
    tools.append(tool_function)
    llm_prompt = f"Add the following mutation to the prompt using the tool: {mutation.value}\n\nPrompt: {prompt['prompt']}"
    # return model.invoke(llm_prompt)
    agent = create_agent(model, tools=tools,
                         response_format=ToolStrategy(BasePrompt),
                         middleware=[mutation_type_prompt],
                         context_schema=Context)
    messages = [{"role": "user", "content": llm_prompt}]
    response = agent.invoke(
        {"messages": messages},
        context={"mutation_type": mutation.value}
    )
    return response["structured_response"]


def invoke_llm_with_tools(state: MutationWorkflowState):
    """2. Invokes LLM (placeholder) to perform the mutation."""
    print(
        f"  [Mutation Subgraph] Invoking LLM to mutate: '{state['prompt_to_mutate']}'")
    mutation = Mutation(state['mutation_type'])
    output: BasePrompt = invoke_llm(
        state['prompt_to_mutate'], mutation)

    return {"mutated_prompt": MutatedPrompt(prompt=output['prompt'], mutation_type=mutation)}
