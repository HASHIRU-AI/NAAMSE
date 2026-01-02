"""
Green Agent Executor - Base class for green agents.
Copied from https://github.com/agentbeats/tutorial/blob/main/src/agentbeats/green_executor.py
"""
from abc import abstractmethod
from pydantic import ValidationError

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidParamsError,
    Task,
    TaskState,
    UnsupportedOperationError,
    InternalError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from src.agentbeats.models import NAAMSERequest


class GreenAgent:
    """
    Abstract base class for Green Agents.
    
    Subclass this and implement:
    - run_eval(): Your evaluation logic
    - validate_request(): Validate incoming requests
    """

    @abstractmethod
    async def run_eval(self, request: NAAMSERequest, updater: TaskUpdater) -> None:
        """Run the evaluation. Override this with your logic."""
        pass

    @abstractmethod
    def validate_request(self, request: NAAMSERequest) -> tuple[bool, str]:
        """Validate the incoming request. Return (ok, message)."""
        pass


class GreenExecutor(AgentExecutor):
    """
    A2A AgentExecutor that wraps a GreenAgent.
    Handles the A2A protocol and delegates to the GreenAgent.
    """

    def __init__(self, green_agent: GreenAgent):
        self.agent = green_agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the green agent evaluation."""
        request_text = context.get_user_input()
        
        try:
            req: NAAMSERequest = NAAMSERequest.model_validate_json(request_text)
            ok, msg = self.agent.validate_request(req)
            if not ok:
                raise ServerError(error=InvalidParamsError(message=msg))
        except ValidationError as e:
            raise ServerError(error=InvalidParamsError(message=e.json()))

        msg = context.message
        if msg:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        else:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting assessment.\n{req.model_dump_json()}",
                context_id=context.context_id
            )
        )

        try:
            await self.agent.run_eval(req, updater)
            await updater.complete()
        except Exception as e:
            print(f"Agent error: {e}")
            await updater.failed(
                new_agent_text_message(f"Agent error: {e}", context_id=context.context_id)
            )
            raise ServerError(error=InternalError(message=str(e)))

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
