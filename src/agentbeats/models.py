"""
Pydantic models for NAAMSE Green Agent input/output.
"""
from typing import Optional
from pydantic import BaseModel, HttpUrl


class ScoredPrompt(BaseModel):
    """A prompt with its score."""
    prompt: list[str]
    score: float = 0.0


class NAAMSERequest(BaseModel):
    """Input configuration for the NAAMSE fuzzer."""
    target_url: HttpUrl 
    iterations_limit: int = 3
    mutations_per_iteration: int = 2
    score_threshold: float = 50.0
    initial_prompts: Optional[list[ScoredPrompt]] = None


