"""
Pydantic models for NAAMSE Green Agent input/output.

Based on AgentBeats tutorial:
https://github.com/RDI-Foundation/agentbeats-tutorial/blob/main/src/agentbeats/models.py
"""
from typing import Any, Optional
from pydantic import BaseModel, HttpUrl


# Standard AgentBeats request format (used by ALL green agents)
# Reference: https://github.com/RDI-Foundation/agentbeats-tutorial/blob/main/src/agentbeats/models.py#L4-L7
class EvalRequest(BaseModel):
    """
    Standard request format from AgentBeats framework.
    
    All green agents receive this format:
    - participants: mapping of role names to agent endpoint URLs
    - config: scenario-specific configuration (interpreted by each green agent)
    """
    participants: dict[str, HttpUrl]  # e.g., {"agent": "http://localhost:5000"}
    config: dict[str, Any]            # NAAMSE-specific config goes here


# Standard AgentBeats result format
# Reference: https://github.com/RDI-Foundation/agentbeats-tutorial/blob/main/src/agentbeats/models.py#L9-L11
class EvalResult(BaseModel):
    """Assessment results returned as artifact."""
    detail: dict[str, Any]


# NAAMSE-specific configuration (extracted from EvalRequest.config)
class NAAMSEConfig(BaseModel):
    """
    Configuration for NAAMSE fuzzer, extracted from EvalRequest.config.
    
    Used in scenario.toml under [config] section.
    """
    iterations_limit: int = 3
    mutations_per_iteration: int = 2
    score_threshold: float = 50.0


