"""
NAAMSE Green Agent Server - Entry point to run the agent.

Based on AgentBeats tutorial:
https://github.com/RDI-Foundation/agentbeats-tutorial/blob/main/scenarios/tau2/tau2_evaluator.py#L467-L491
"""
# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from .executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="fuzz-agent",
        name="Fuzz LLM Agent Security",
        description=(
            "Run the NAAMSE (Neural Adversarial Agent Mutation-based Security Evaluator) fuzzer "
            "against a target agent to discover vulnerabilities through iterative prompt mutations."
        ),
        tags=["evaluation", "llm-security", "fuzzing", "adversarial"],
        examples=[
            '''{\n  "participants": {\n    "agent": "http://localhost:5000"\n  },\n  "config": {\n    "iterations_limit": 5,\n    "mutations_per_iteration": 3,\n    "score_threshold": 50.0\n  }\n}'''
        ]
    )

    agent_card = AgentCard(
        name="NAAMSE Fuzzer",
        description=(
            "NAAMSE (Neural Adversarial Agent Mutation-based Security Evaluator) - "
            "An LLM security fuzzer that evaluates agent safety through iterative prompt mutations and scoring."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
