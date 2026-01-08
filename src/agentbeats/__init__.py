"""
NAAMSE Green Agent - A2A agent for fuzzing LLM agents.
"""
# Load environment variables before importing anything that uses them
from dotenv import load_dotenv
load_dotenv()

from .agent import Agent
from .executor import Executor
from .models import EvalRequest, NAAMSEConfig, EvalResult

__all__ = [
    "Agent",
    "Executor",
    "EvalRequest",
    "NAAMSEConfig",
    "EvalResult",
]
