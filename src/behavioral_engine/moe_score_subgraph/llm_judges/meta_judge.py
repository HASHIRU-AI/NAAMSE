from src.behavioral_engine.moe_score_subgraph.llm_judges.llm_judge import LLMJudge
from src.behavioral_engine.moe_score_subgraph.moe_score_judge_prompts import EvalType

from langchain_openai import ChatOpenAI

META_MODEL_API_BASE_URL = "https://api.meta.ai/v1"


class _AutoToolChoiceChatOpenAI(ChatOpenAI):
    """ChatOpenAI that downgrades forced tool choices to "auto".

    Meta Model API rejects any tool_choice other than "auto", but create_agent's
    ToolStrategy forces tool_choice="any" to get structured output.
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice not in (None, "auto"):
            tool_choice = "auto"
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)


class MetaJudge(LLMJudge):
    """Meta Muse Spark judge implementation via the OpenAI-compatible Meta Model API"""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0,
        judge_id: str = None,
        eval_type: EvalType = None,
        base_url: str = None,
    ):
        """
        Initialize Meta judge.

        Args:
            model_name: Meta Model API model ID (e.g., "muse-spark-1.2")
            temperature: Sampling temperature for model generation
            judge_id: Unique identifier for this judge instance
            eval_type: Evaluation type for this judge
            base_url: Override for the Meta Model API base URL
        """
        import os
        self.model_name = model_name or os.getenv("META_JUDGE_MODEL", "muse-spark-1.2")
        self.temperature = temperature
        self.base_url = base_url or os.getenv("META_MODEL_API_BASE_URL", META_MODEL_API_BASE_URL)
        self._judge_id = judge_id or f"meta_{self.model_name.replace('.', '_').replace('-', '_')}"
        self.system_prompt = None
        self.eval_type = eval_type

    def get_model(self):
        """Return a ChatOpenAI client pointed at the Meta Model API."""
        import os
        return _AutoToolChoiceChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
            api_key=os.getenv("MODEL_API_KEY"),
        )

    def get_name(self) -> str:
        return f"Meta-{self._judge_id}"

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
