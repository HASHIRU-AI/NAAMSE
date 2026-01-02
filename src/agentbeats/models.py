"""
Pydantic models for NAAMSE Green Agent input/output.
"""
from typing import Optional
from pydantic import BaseModel, HttpUrl

from mutation_engine.mutation_workflow_state import ScoredPrompt


class NAAMSERequest(BaseModel):
    """Input configuration for the NAAMSE fuzzer."""
    target_url: HttpUrl 
    iterations_limit: int = 3
    mutations_per_iteration: int = 2
    score_threshold: float = 50.0
    initial_prompts: Optional[list[ScoredPrompt]] = None


