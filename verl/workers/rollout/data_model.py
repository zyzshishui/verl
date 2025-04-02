from enum import Enum
from typing import List
from pydantic import BaseModel

from verl.workers.tool.data_model import OpenAIFunctionToolSchema


class Message(BaseModel):
    role: str
    content: str


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""
    request_id: str
    state: AsyncRolloutRequestStateEnum
    prompt: str
    messages: List[Message]
    tools: List[OpenAIFunctionToolSchema]
