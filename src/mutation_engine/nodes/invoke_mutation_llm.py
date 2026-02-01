from typing import Any, Dict
from typing_extensions import TypedDict
from typing_extensions import TypedDict
from src.cluster_engine.utilities import get_db
from src.cluster_engine.data_access.data_source import DataSource
from src.mutation_engine.mutation_workflow_state import BasePrompt, Metadata, MutatedPrompt, Mutation, MutationWorkflowState
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from src.helpers.extract_text_from_context import extract_text_from_content
import json  # Added for json.loads in invoke_llm_with_tools
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableConfig


class Context(TypedDict):
    mutation_type: Mutation
    task_seed: int


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
        api_key = os.getenv("MUTATION_ENGINE_API_KEY") or os.getenv(
            "GOOGLE_API_KEY")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
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


def invoke_llm(prompt: BasePrompt, mutation: Mutation, task_seed: int, database: DataSource) -> BasePrompt:
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
            context={"mutation_type": mutation.value,
                     "task_seed": task_seed},
            config={
                "recursion_limit": 10,  # Maximum 10 iterations
                "configurable": {
                    "thread_id": "mutation_thread"
                }
            }
        )

        # If tools were used, extract result from the tool call
        if tools:
            # Validate response structure
            if not response or "messages" not in response:
                raise ValueError(
                    f"Invalid agent response structure. Keys: {response.keys() if response else 'None'}")

            if not response["messages"]:
                raise ValueError(
                    "Agent returned empty messages array - LLM may have refused or failed to respond")

            print(
                f"  [DEBUG] Extracting result from {len(response['messages'])} messages")

            # Strategy 1: Look for ToolMessage (the tool's actual return value)
            for i, msg in enumerate(reversed(response["messages"])):
                msg_type = type(msg).__name__
                print(f"  [DEBUG] Message {i}: type={msg_type}")

                # Check if this is a ToolMessage from our mutation tool
                if hasattr(msg, 'name') and msg.name in [tool.name for tool in tools]:
                    print(f"  [DEBUG] Found ToolMessage from {msg.name}")
                    content = msg.content

                    # Handle different content types
                    # Case 1: Content is already a dict (direct return)
                    if isinstance(content, dict):
                        if "prompt" in content:
                            print(
                                f"  [DEBUG] Tool returned dict directly with 'prompt' key")
                            return content
                        else:
                            print(
                                f"  [ERROR] Tool returned dict but missing 'prompt' key. Keys: {content.keys()}")

                    # Case 2: Content is a JSON string
                    elif isinstance(content, str):
                        try:
                            import json
                            parsed = json.loads(content)
                            if isinstance(parsed, dict) and "prompt" in parsed:
                                print(
                                    f"  [DEBUG] Tool returned JSON string, parsed successfully")
                                return parsed
                        except json.JSONDecodeError as e:
                            print(
                                f"  [DEBUG] Content is string but not valid JSON: {e}")
                            # Treat as literal prompt text
                            return {"prompt": [content]}

                    # Case 3: Content is a list (treat as prompt list)
                    elif isinstance(content, list):
                        print(
                            f"  [DEBUG] Tool returned list, wrapping as prompt")
                        return {"prompt": content}

                    # Case 4: Other types - convert to string
                    else:
                        print(
                            f"  [DEBUG] Tool returned unexpected type {type(content)}, converting to string")
                        return {"prompt": [str(content)]}

            # Strategy 2: No ToolMessage found - LLM may have responded directly
            # This happens if the LLM doesn't call the tool
            print(
                f"  [WARNING] No ToolMessage found - LLM may not have called the tool")

            if not response["messages"]:
                raise ValueError(
                    "No messages available to extract response from")

            last_message = response["messages"][-1]
            print(
                f"  [DEBUG] Checking last message: type={type(last_message).__name__}")

            # Try to extract any content from the last message
            content = None
            if hasattr(last_message, 'content'):
                content = last_message.content
            elif hasattr(last_message, 'text'):
                content = last_message.text

            if content:
                print(
                    f"  [DEBUG] Extracted content from last message: {str(content)[:200]}")
                if isinstance(content, str):
                    return {"prompt": [content]}
                elif isinstance(content, list):
                    return {"prompt": content}
                elif isinstance(content, dict) and "prompt" in content:
                    return content
                else:
                    return {"prompt": [str(content)]}

            # Strategy 3: Complete failure - provide diagnostic info
            print(f"  [ERROR] Could not extract any content from response")
            print(
                f"  [ERROR] Last message type: {type(last_message).__name__}")
            print(
                f"  [ERROR] Last message attributes: {[a for a in dir(last_message) if not a.startswith('_')]}")
            raise ValueError(
                f"LLM did not call tool '{tools[0].name}' and returned no usable content. "
                f"This likely means the LLM refused or failed to process the request."
            )

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
        raise  # Re-raise the exception instead of returning None


def invoke_llm_with_tools(state: MutationWorkflowState, config: RunnableConfig):
    """2. Invokes LLM (placeholder) to perform the mutation."""
    print(
        f"  [Mutation Subgraph] Invoking LLM to mutate: '{state['prompt_to_mutate']}'")
    mutation = Mutation(state['mutation_type'])
    task_seed = state.get('task_seed', None)
    output: BasePrompt
    database = get_db(config)

    load_dotenv()  # Load environment variables from .env file

    skip_llm = os.getenv("SKIP_LLM", "false").lower() == "true"

    try:

        if skip_llm:
            print(
                f"  [Mutation Subgraph] Skipping LLM invocation as per configuration. Returning original prompt.")
            output = state['prompt_to_mutate']
        else:
            output = invoke_llm(
                state['prompt_to_mutate'], mutation, task_seed=task_seed, database=database)

        # sanitize output to make sure output follows openai format
        sanitized_output = []
        for msg in output['prompt']:
            # Build a new dictionary with only the standard keys
            clean_msg = extract_text_from_content(msg)

            sanitized_output.append(clean_msg)

        # merge sanitized output to single string
        merged_output = " ".join(sanitized_output)
        output['prompt'] = [merged_output]

    except Exception as e:
        print(f"  [ERROR] LLM invocation failed: {e}")
        # keep original prompt in case of failure
        output = state['prompt_to_mutate']

    # Create metadata for the mutated prompt
    metadata: Metadata = {"mutation_type": mutation}
    if 'metadata' in state['prompt_to_mutate'] and state['prompt_to_mutate']['metadata'] and 'cluster_info' in state['prompt_to_mutate']['metadata']:
        metadata['cluster_info'] = state['prompt_to_mutate']['metadata']['cluster_info']

    # Append parent details to history
    history = state['prompt_to_mutate'].get('metadata', {}).get('history', [])
    history = history.copy()  # make a copy to avoid mutating original
    history.append({
        "mutation_type": state['prompt_to_mutate'].get('metadata', {}).get('mutation_type', Mutation.EXPLORE),
        "score": state['prompt_to_mutate'].get('score', 0.0)
    })
    metadata["history"] = history

    mutated = MutatedPrompt(
        prompt=output['prompt'],
        metadata=metadata
    )

    result = {"mutated_prompt": mutated}
    print(f"  [DEBUG] Returning result: {result}")
    return result
