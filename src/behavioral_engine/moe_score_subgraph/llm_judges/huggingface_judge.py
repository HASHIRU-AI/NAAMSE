from src.behavioral_engine.moe_score_subgraph.llm_judges.llm_judge import LLMJudge
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


class HuggingFaceJudge(LLMJudge):
    """HuggingFace-based judge implementation using local models"""
    
    def __init__(
        self, 
        model_id: str = "google/gemma-3-270m-it",
        temperature: float = 0,
        judge_id: str = None,
        eval_type: EvalType = None,
        device: int = None,
        max_new_tokens: int = 512,
        **model_kwargs
    ):
        """
        Initialize HuggingFace judge with local model pipeline.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "google/gemma-3-270m-it")
            temperature: Sampling temperature for model generation
            judge_id: Unique identifier for this judge instance
            eval_type: Evaluation type for this judge
            device: Device to run model on (None for auto-detect, -1 for CPU, 0+ for GPU)
            max_new_tokens: Maximum number of tokens to generate
            **model_kwargs: Additional keyword arguments for pipeline configuration
        """
        self.model_id = model_id
        self.temperature = temperature
        # Create judge_id from model_id if not provided
        self._judge_id = judge_id or f"hf_{model_id.split('/')[-1].replace('.', '_').replace('-', '_')}"
        self.system_prompt = None
        self.eval_type = eval_type
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_kwargs = model_kwargs
    
    def get_model(self):
        """Initialize and return the HuggingFace chat model with structured output support"""
        import os
        
        # Get HuggingFace token if available
        token = os.getenv("HF_TOKEN")
        
        # Configure pipeline kwargs (parameters for the pipeline itself)
        pipeline_kwargs = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "token" : token,
        }
        
        # Configure model kwargs (parameters for loading the model)
        model_kwargs = self.model_kwargs.copy() if self.model_kwargs else {}
        
        # Create the HuggingFace pipeline using from_model_id
        llm = HuggingFacePipeline.from_model_id(
            model_id=self.model_id,
            task="text-generation",
            device=self.device,
            pipeline_kwargs=pipeline_kwargs,
            model_kwargs=model_kwargs,
        )
        
        # Wrap in ChatHuggingFace for structured output support
        return ChatHuggingFace(llm=llm)
    
    def get_name(self) -> str:
        return f"HuggingFace-{self._judge_id}"
    
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
