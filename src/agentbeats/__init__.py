"""
NAAMSE Green Agent - A2A agent for fuzzing LLM agents.
"""
# Load environment variables before importing anything that uses them
from dotenv import load_dotenv
load_dotenv()

from src.agentbeats.agent import Agent
from src.agentbeats.executor import Executor
from src.agentbeats.models import EvalRequest, NAAMSEConfig, EvalResult

__all__ = [
    "Agent",
    "Executor",
    "EvalRequest",
    "NAAMSEConfig",
    "EvalResult",
]
