"""
NAAMSE Green Agent - A2A agent for fuzzing LLM agents.
"""
# Load environment variables before importing anything that uses them
from dotenv import load_dotenv
load_dotenv()

from src.agentbeats.models import NAAMSERequest
from src.agentbeats.naamse_green_agent import NAAMSEGreenAgent

__all__ = [
    "NAAMSERequest",
    "NAAMSEGreenAgent",
]
