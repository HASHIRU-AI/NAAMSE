"""
NAAMSE Green Agent - Evaluator agent that runs the NAAMSE LangGraph fuzzer.

Based on AgentBeats template:
https://github.com/RDI-Foundation/agentbeats-tutorial

This green agent evaluates purple agents by running adversarial prompt mutations
and scoring their responses.
"""
import json
from enum import Enum
from pydantic import ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

# Import the NAAMSE fuzzer graph
from src.agent.graph import graph

# Import models from models.py to avoid duplication
from src.agentbeats.models import EvalRequest, NAAMSEConfig


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum types."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class Agent:
    """
    NAAMSE Green Agent - Security fuzzer for LLM agents.
    
    Runs adversarial prompt mutations against a target agent and
    scores the responses to find vulnerabilities.
    """
    
    # Required participant: the target agent to test
    required_roles: list[str] = ["agent"]
    
    # Required config keys (all have defaults, so none strictly required)
    required_config_keys: list[str] = []

    def __init__(self):
        # NAAMSE doesn't need a Messenger since the fuzzer graph handles
        # communication with the target agent internally via a2a_agent_url
        pass

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the incoming assessment request."""
        # Check required participant roles
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing participant roles: {missing_roles}. Expected: {self.required_roles}"

        # Validate target URL
        target_url = str(request.participants["agent"])
        if not target_url.startswith(("http://", "https://")):
            return False, f"Invalid target URL: {target_url}"

        # Validate NAAMSE config structure
        try:
            config = NAAMSEConfig(**request.config)
            if config.iterations_limit < 1:
                return False, "iterations_limit must be at least 1"
            if config.mutations_per_iteration < 1:
                return False, "mutations_per_iteration must be at least 1"
        except Exception as e:
            return False, f"Invalid NAAMSE config: {e}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run the NAAMSE fuzzer assessment.
        
        Args:
            message: The incoming A2A message containing EvalRequest
            updater: TaskUpdater to report progress and results
        """
        input_text = get_message_text(message)

        # Parse and validate request
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Extract target URL and config
        target_url = str(request.participants["agent"])
        config = NAAMSEConfig(**request.config)

        # Report assessment start
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"🚀 Starting NAAMSE Security Assessment\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Target: {target_url}\n"
                f"Iterations: {config.iterations_limit}\n"
                f"Mutations per iteration: {config.mutations_per_iteration}\n"
                f"Score threshold: {config.score_threshold}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        )

        # Build fuzzer input state
        initial_prompts = config.initial_prompts if config.initial_prompts else [
            {"prompt": ["seed prompt"], "score": 0.0}
        ]

        fuzzer_input = {
            "iterations_limit": config.iterations_limit,
            "mutations_per_iteration": config.mutations_per_iteration,
            "score_threshold": config.score_threshold,
            "a2a_agent_url": target_url,
            "input_prompts_for_iteration": initial_prompts
        }

        try:
            # Progress update
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("⚙️ Running fuzzer iterations...")
            )

            # Run the NAAMSE fuzzer graph
            final_state = await graph.ainvoke(fuzzer_input)

        except Exception as e:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"❌ Fuzzer error: {str(e)}")
            )
            raise

        # Extract report from fuzzer state
        report_result = final_state.get("report", {})

        # Progress update
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("📊 Generating assessment report...")
        )

        # Log to terminal
        print("\n" + "=" * 60)
        print("NAAMSE Fuzzer Assessment Report")
        print("=" * 60)
        print(json.dumps(report_result, indent=2, cls=EnumEncoder))
        print("=" * 60 + "\n")

        # Return structured results as artifact
        # Reference: https://github.com/RDI-Foundation/agentbeats-tutorial/blob/main/scenarios/tau2/tau2_evaluator.py#L362-L367
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=json.dumps(report_result, indent=2, cls=EnumEncoder))),
                Part(root=DataPart(data=report_result)),  # Structured JSON for programmatic access
            ],
            name="NAAMSE Assessment Results",
        )