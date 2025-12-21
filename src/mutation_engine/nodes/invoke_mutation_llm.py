from typing_extensions import TypedDict
from src.mutation_engine.mutation_workflow_state import BasePrompt, MutatedPrompt, Mutation, MutationWorkflowState
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ModelRequest, dynamic_prompt


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


# Cache for agents - key is tuple of tool names
_agent_cache = {}


def get_or_create_agent(tools: list):
    """Get cached agent or create new one if not exists."""
    # Create a cache key based on tool names
    tool_names = tuple(
        sorted([tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]))

    if tool_names not in _agent_cache:
        import os
        api_key = os.getenv("MUTATION_ENGINE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            safety_settings={
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
        })

        # If we have tools, don't use ToolStrategy for structured output
        # Let the agent call tools naturally and extract result from final message
        if tools:
            agent = create_agent(
                model,
                tools=tools,
                middleware=[mutation_type_prompt],
                context_schema=Context
            )
        else:
            # No tools - use structured output directly
            agent = create_agent(
                model,
                tools=tools,
                response_format=ToolStrategy(BasePrompt),
                middleware=[mutation_type_prompt],
                context_schema=Context
            )
        _agent_cache[tool_names] = agent
        print(f"  [DEBUG] Agent cached with key: {tool_names}")
    else:
        print(f"  [DEBUG] Using cached agent for tools: {tool_names}")

    return _agent_cache[tool_names]


def invoke_llm(prompt: BasePrompt, mutation: Mutation) -> BasePrompt:
    """1. Invokes LLM with optional tools to perform the mutation."""
    tools = []
    llm_prompt = f"Add the following mutation to the prompt: Mutation: {mutation.value}\n\nPrompt: {prompt['prompt']}"
    try:
        # for mutation in Mutation:
        module_name = mutation.value
        module = __import__(
            f"src.mutation_engine.nodes.mutations.{module_name}", fromlist=[''])
        tool_function = getattr(module, module_name)
        tools.append(tool_function)
    except Exception as e:
        print(
            f"  [Mutation Subgraph] Could not load tool for mutation type: {mutation.value}. Using direct LLM invocation.")
        print(e)

    # Get or create cached agent
    agent = get_or_create_agent(tools)

    messages = [{"role": "user", "content": llm_prompt}]

    try:
        response = agent.invoke(
            {"messages": messages},
            context={"mutation_type": mutation.value},
            config={
                "recursion_limit": 10,  # Maximum 10 iterations
                "configurable": {
                    "thread_id": "mutation_thread"
                }
            }
        )

        # If tools were used, extract result from the last tool message or AI message
        if tools:
            # Look for the last tool message result
            for msg in reversed(response["messages"]):
                if hasattr(msg, 'name') and msg.name in [tool.name for tool in tools]:
                    # This is a tool message with the result
                    import json
                    try:
                        result = json.loads(msg.content)
                        if isinstance(result, dict) and "prompt" in result:
                            return result
                    except:
                        pass

            # Fallback: look at final AI message
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                print(
                    f"  [DEBUG] Using final AI message content: {last_message.content}")
                return {"prompt": [last_message.content]}

            raise ValueError(
                "Could not extract mutated prompt from tool calls")

        # No tools - check for structured_response
        if "structured_response" in response:
            return response["structured_response"]
        else:
            print(
                f"  [ERROR] No structured_response in agent response. Keys: {response.keys()}")
            print(f"  [ERROR] Full response: {response}")
            # Fallback: try to extract prompt from messages
            if "messages" in response and len(response["messages"]) > 0:
                last_message = response["messages"][-1]
                print(f"  [DEBUG] Last message: {last_message}")
                # Try to construct a BasePrompt from the last message
                if hasattr(last_message, 'content'):
                    return {"prompt": [last_message.content]}
            raise ValueError("Agent did not return structured_response")

    except Exception as e:
        print(f"  [ERROR] Exception during agent invocation: {e}")
        print(f"  [ERROR] Exception type: {type(e)}")


def invoke_llm_with_tools(state: MutationWorkflowState):
    """2. Invokes LLM (placeholder) to perform the mutation."""
    print(
        f"  [Mutation Subgraph] Invoking LLM to mutate: '{state['prompt_to_mutate']}'")
    mutation = Mutation(state['mutation_type'])
    output: BasePrompt = invoke_llm(
        state['prompt_to_mutate'], mutation)
    result = {"mutated_prompt": MutatedPrompt(
        prompt=output['prompt'], mutation_type=mutation)}
    print(f"  [DEBUG] Returning result: {result}")
    return result
