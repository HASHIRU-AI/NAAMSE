"""
NAAMSE Green Agent Server - Entry point to run the agent.
"""
# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
load_dotenv()

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.agentbeats.naamse_green_agent import NAAMSEGreenAgent
from src.agentbeats.green_executor import GreenExecutor


def create_agent_card(host: str, port: int) -> AgentCard:
    """Create the A2A agent card for NAAMSE."""
    return AgentCard(
        name="NAAMSE Fuzzer",
        description="NAAMSE Evaluator",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=[
            AgentSkill(
                id="fuzz-agent",
                name="Fuzz Agent",
                description="Run the NAAMSE fuzzer against a target agent to discover vulnerabilities. "
                            "Provide a JSON request with 'target_url' and optional fuzzer config.",
                tags=["evaluation", "llm-security"],
            )
        ],
    )


def create_app(host: str, port: int) -> A2AStarletteApplication:
    """Create the A2A application."""

    green_agent = NAAMSEGreenAgent()
    executor = GreenExecutor(green_agent)
    
    agent_card = create_agent_card(host, port)
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )


@click.command()
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def main(host: str, port: int):
    """Run the NAAMSE Green Agent server."""
    import uvicorn
    
    print(f"Starting NAAMSE Green Agent on http://{host}:{port}")
    print(f"Agent card available at: http://{host}:{port}/.well-known/agent.json")
    
    app = create_app(host, port)
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    main()
