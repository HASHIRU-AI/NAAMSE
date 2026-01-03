"""
NAAMSE Green Agent - Evaluator agent that calls the NAAMSE LangGraph fuzzer.
"""
import json
from enum import Enum

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from src.agentbeats.models import NAAMSERequest
from src.agentbeats.green_executor import GreenAgent
from src.agent.graph import graph


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum types."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class NAAMSEGreenAgent(GreenAgent):
    """
    NAAMSE Green Agent implementation.

    Receives a NAAMSERequest, runs the fuzzer, returns result.
    """

    def validate_request(self, request: NAAMSERequest) -> tuple[bool, str]:
        """Validate the incoming request."""
        if not str(request.target_url).startswith(("http://", "https://")):
            return False, f"Invalid target URL: {request.target_url}"

        if request.iterations_limit < 1:
            return False, "iterations_limit must be at least 1"

        return True, "Request is valid"

    async def run_eval(self, request: NAAMSERequest, updater: TaskUpdater) -> None:
        """Run the NAAMSE fuzzer evaluation."""
        target_url = str(request.target_url)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Running NAAMSE fuzzer against {target_url} with {request.iterations_limit} iterations...",
                context_id=updater.context_id
            )
        )

        # Build the fuzzer input state
        initial_prompts = [p.model_dump() for p in request.initial_prompts] if request.initial_prompts else [
            {"prompt": ["seed prompt"], "score": 0.0}
        ]

        fuzzer_input = {
            "iterations_limit": request.iterations_limit,
            "mutations_per_iteration": request.mutations_per_iteration,
            "score_threshold": request.score_threshold,
            "a2a_agent_url": target_url,
            "input_prompts_for_iteration": initial_prompts
        }

        try:
            final_state = await graph.ainvoke(fuzzer_input)
        except Exception as e:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Fuzzer error: {e}", context_id=updater.context_id)
            )
            raise

        report_result = final_state.get("report", {})

        # Pretty print for terminal readability
        print(json.dumps(report_result, indent=2, cls=EnumEncoder))

        # Send final result
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(json.dumps(
                report_result, indent=2, cls=EnumEncoder), context_id=updater.context_id)
        )
