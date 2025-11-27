from src.behavioral_engine.moe_score_subgraph.llm_judges.llm_judge import LLMJudge
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType

from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory


class GeminiJudge(LLMJudge):
    """Gemini-based judge implementation"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0, judge_id: str = None, eval_type: EvalType = None):
        self.model_name = model_name
        self.temperature = temperature
        self._judge_id = judge_id or f"gemini_{model_name.replace('.', '_').replace('-', '_')}"
        self.system_prompt = None
        self.eval_type = eval_type
    
    def get_model(self):
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.OFF,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.OFF,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.OFF,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.OFF,
            }
        )
    
    def get_name(self) -> str:
        return f"Gemini-{self._judge_id}"
    
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