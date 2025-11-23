import asyncio
from python_a2a import A2AClient, Message, TextContent, MessageRole
from typing import List, Any, Dict

def _sync_a2a_interaction(prompts: List[str], agent_url: str) -> Dict[str, Any]:
    """
    Helper function that runs the blocking (synchronous) A2A client code 
    in a separate thread so it doesn't freeze the async event loop.
    """
    # Initialize client synchronously
    client = A2AClient(agent_url)

    conversation_history = []
    temp_task_id = None

    for prompt_text in prompts:
        # Create message
        outbound_message = Message(
            role=MessageRole.USER,
            content=TextContent(text=prompt_text)
        )

        # Send message (Synchronous call - NO 'await')
        if temp_task_id is None:
            # First turn: create task implicitly or explicitly depending on SDK version
            # Using send_message is generally safer for both cases in newer SDKs
            response = client.send_message(message=outbound_message)
            
            # Capture ID for context
            temp_task_id = getattr(response, "task_id", None) or getattr(response, "context_id", None)
        else:
            # Subsequent turns: pass ID
            response = client.send_message(
                message=outbound_message,
                task_id=temp_task_id
            )

        # Append to history
        conversation_history.append({"role": "user", "content": prompt_text})
        
        response_text = "No response text"
        if hasattr(response.content, "text"):
            response_text = response.content.text
            
        conversation_history.append({"role": "assistant", "content": response_text})

    return {"conversation_history": {"messages": conversation_history}}

async def invoke_a2a_agent(state: Dict[str, Any]):
    """
    Main Async Node: Offloads the blocking A2A interaction to a thread.
    """
    prompts: List[str] = state["prompt"]["prompt"]
    agent_url: str = state["a2a_agent_url"]

    # Run the synchronous helper in a separate thread to satisfy LangGraph's async checks
    return await asyncio.to_thread(_sync_a2a_interaction, prompts, agent_url)