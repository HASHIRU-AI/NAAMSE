from src.behavioral_engine.moe_score_subgraph.llm_judges.llm_judge import LLMJudge
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType

from langchain_ollama import ChatOllama


class OllamaJudge(LLMJudge):
    """Ollama-based judge implementation using local models via Ollama server.
    
    Ollama handles concurrency natively - multiple judges can make parallel requests
    to the same model without memory issues since Ollama manages the model instance.
    """
    
    def __init__(
        self, 
        model: str = "llama3.2:3b", 
        temperature: float = 0, 
        judge_id: str = None, 
        eval_type: EvalType = None,
    ):
        """
        Initialize Ollama judge.
        
        Args:
            model: Ollama model name (e.g., "gemma3:270m", "llama3.2", "mistral")
            temperature: Sampling temperature for model generation
            judge_id: Unique identifier for this judge instance
            eval_type: Evaluation type for this judge
        """
        self.model = model
        self.temperature = temperature
        self._judge_id = judge_id or f"ollama_{model.replace(':', '_').replace('.', '_').replace('-', '_')}"
        self.system_prompt = None
        self.eval_type = eval_type
    
    def get_model(self):
        """Return ChatOllama instance with structured output support."""
        return ChatOllama(
            model=self.model,
            temperature=self.temperature,
        )
    
    def get_name(self) -> str:
        return f"Ollama-{self._judge_id}"
    
    def get_judge_id(self) -> str:
        return self._judge_id
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for this judge, if any"""
        return self.system_prompt
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for this judge"""
        self.system_prompt = prompt
        
    def get_eval_type(self) -> EvalType:
        """Return the eval type for this judge based on judge_id"""
        return self.eval_type
