"""
Test script for the NAAMSE Green Agent.

Run the server first:
    .venv\Scripts\python.exe -m src.agentbeats.server --host 127.0.0.1 --port 9009

Then run this test:
    .venv\Scripts\python.exe src/agentbeats/test_green_agent.py --target http://localhost:5000
"""
import asyncio
import json
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


async def test_naamse_agent(
    target_url: str = "http://localhost:5000",
    green_agent_url: str = "http://localhost:8000",
    iterations_limit: int = 1,
    mutations_per_iteration: int = 1,
):
    """Send a test request to the NAAMSE Green Agent."""

    # Build EvalRequest payload
    eval_request = {
        "participants": {
            "agent": target_url
        },
        "config": {
            "iterations_limit": iterations_limit,
            "mutations_per_iteration": mutations_per_iteration,
            "score_threshold": 50.0,
        }
    }

    print(f"🚀 Testing NAAMSE Green Agent")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Green Agent: {green_agent_url}")
    print(f"Target Agent: {target_url}")
    print(f"Config:")
    print(f"  - Iterations: {iterations_limit}")
    print(f"  - Mutations per iteration: {mutations_per_iteration}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    async with httpx.AsyncClient(timeout=300.0) as httpx_client:
        # Resolve agent card
        resolver = A2ACardResolver(
            httpx_client=httpx_client, base_url=green_agent_url)
        agent_card = await resolver.get_agent_card()

        print(f"✅ Connected to: {agent_card.name}")
        print(f"   Description: {agent_card.description}\n")

        # Create A2A client
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        # Create message with EvalRequest payload
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(eval_request)))],
            message_id=uuid4().hex,
            context_id=None,
        )

        print("📡 Sending evaluation request...\n")

        # Stream events from the agent
        task_completed = False
        async for event in client.send_message(msg):
            print("----- Event Received -----")
            match event:
                case Message() as response_msg:
                    # Extract text from message parts
                    for part in response_msg.parts:
                        if hasattr(part.root, 'text'):
                            print(f"💬 Agent: {part.root.text}\n")

                case (task, update):
                    # Task status update
                    status = task.status
                    if status.message:
                        for part in status.message.parts:
                            if hasattr(part.root, 'text'):
                                print(f"📊 Status: {part.root.text}")

                    # Check for artifacts (final results)
                    if task.artifacts:
                        print(f"\n🎯 Assessment Complete!\n")
                        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        for artifact in task.artifacts:
                            for part in artifact.parts:
                                if hasattr(part.root, 'text'):
                                    print(part.root.text)
                                elif hasattr(part.root, 'data'):
                                    print(json.dumps(part.root.data, indent=2))
                        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                    # Check if task is complete
                    if task.status.state.value in ['completed', 'failed', 'rejected']:
                        print(f"✅ Task {task.status.state.value}\n")
                        task_completed = True

        # Ensure we processed the task
        if not task_completed:
            print("⚠️ Task did not complete as expected")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the NAAMSE Green Agent")
    parser.add_argument(
        "--target", default="http://localhost:5000", help="Target agent URL to fuzz")
    parser.add_argument(
        "--green-agent", default="http://localhost:8000", help="Green agent URL")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of fuzzer iterations")
    parser.add_argument("--mutations", type=int, default=1,
                        help="Mutations per iteration")

    args = parser.parse_args()

    asyncio.run(
        test_naamse_agent(
            target_url=args.target,
            green_agent_url=args.green_agent,
            iterations_limit=args.iterations,
            mutations_per_iteration=args.mutations,
        )
    )
