from src.behavioral_engine.moe_score_subgraph.llm_judges.llm_judge import LLMJudge
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType

from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory


class GeminiJudge(LLMJudge):
    """Gemini-based judge implementation"""
    
    def __init__(self, model_name: str = None, temperature: float = 0, judge_id: str = None, eval_type: EvalType = None):
        import os
        self.model_name = model_name or os.getenv("BEHAVIORAL_ENGINE_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self._judge_id = judge_id or f"gemini_{self.model_name.replace('.', '_').replace('-', '_')}"
        self.system_prompt = None
        self.eval_type = eval_type
    
    def get_model(self):
        import os
        kwargs = {}
        use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")

        if use_vertexai:
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                try:
                    import google.auth
                    # Request the correct scopes for Vertex AI
                    scopes = ['https://www.googleapis.com/auth/cloud-platform']
                    credentials, project = google.auth.default(scopes=scopes)
                    kwargs["credentials"] = credentials
                    if project:
                        kwargs["project"] = project
                except Exception as e:
                    print(f"Warning: Failed to load Vertex credentials: {e}")
        else:
            api_key = os.getenv("BEHAVIORAL_ENGINE_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                kwargs["google_api_key"] = api_key

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            **kwargs,
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