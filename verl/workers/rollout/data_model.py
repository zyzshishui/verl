from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from transformers import PreTrainedTokenizer
from verl.workers.tool.data_model import OpenAIFunctionToolSchema, OpenAIFunctionToolCall
from verl.workers.rollout.utils import compute_position_id_with_mask

class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""
    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")

class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None

class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""
    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tools: Optional[List[OpenAIFunctionToolSchema]] = None
    input_ids: List[int]
    attention_mask: List[int]
    position_ids: List[int]
    loss_mask: List[int]

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> str:
        return tokenizer.apply_chat_template( # type: ignore
            conversation=[msg.model_dump() for msg in self.messages], 
            tools=[tool.model_dump() for tool in self.tools] if self.tools else None, 
            add_generation_prompt=True, 
            tokenize=False
        )

    def add_assistant_message(
        self, 
        tokenizer: PreTrainedTokenizer, 
        content: str, 
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None
    ) -> None:
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        msg_w_chat_template = tokenizer.apply_chat_template(
            conversation=[msg.model_dump()],
            tools=[tool.model_dump() for tool in self.tools] if self.tools else None,
            add_generation_prompt=True,
            tokenize=False
        )
        _input_data = tokenizer(msg_w_chat_template, return_tensors="pt", add_special_tokens=False)
        _input_ids = _input_data["input_ids"][0].tolist()
        _attention_mask = _input_data["attention_mask"][0].tolist()
        _delta_position_ids = compute_position_id_with_mask(_attention_mask).tolist()
        last_position_id = self.position_ids[-1]
        _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
        self.input_ids += _input_ids
        self.attention_mask += _attention_mask
        self.loss_mask += [1] * len(_input_ids)
        self.position_ids += _position_ids
        self.messages.append(msg)
