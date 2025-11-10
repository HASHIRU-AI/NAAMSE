from abc import ABC, abstractmethod

class LLMJudge(ABC):
    """Abstract base class for LLM judges"""
    
    @abstractmethod
    def get_model(self):
        """Return the LLM model instance"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a name for this judge"""
        pass
    
    @abstractmethod
    def get_judge_id(self) -> str:
        """Return a unique identifier for this judge (used for node naming)"""
        pass
