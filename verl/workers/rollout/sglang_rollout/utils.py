import uuid
import torch
from typing import List
from verl.workers.rollout.data_model import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    Message,
)
from verl.workers.tool.data_model import OpenAIFunctionToolSchema
from verl import DataProto


def prompts_to_async_rollout_requests(
    prompts: DataProto, tokenizer, tool: List[OpenAIFunctionToolSchema] = None
) -> List[AsyncRolloutRequest]:
    if tools is None:
        tools = []
    requests = []

    input_ids = prompts.batch["input_ids"]
    batch_size = input_ids.size(0)

    for i in range(batch_size):
        request_id = str(uuid.uuid4())

        if tokenizer is not None:
            prompt_ids = input_ids[i].tolist()
            pad_token_id = (
                tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None
            )
            if pad_token_id is not None:
                prompt_ids = [id for id in prompt_ids if id != pad_token_id]
            prompt = tokenizer.decode(prompt_ids)
        else:
            prompt = str(input_ids[i].tolist())

        messages = [Message(role="user", content=prompt)]

        request = AsyncRolloutRequest(
            request_id=request_id,
            state=AsyncRolloutRequestStateEnum.PENDING,
            prompt=prompt,
            messages=messages,
            tools=tools,
        )

        requests.append(request)

    return requests


def messages_to_ids_with_loss_mask(
    messages: List[Message],
    tokenizer,
    tools: List[OpenAIFunctionToolSchema] = None,
    max_length: int = None,
) -> Tuple[List[int], List[int]]:
    formatted_messages = [
        {"role": msg.role, "content": msg.content} for msg in messages
    ]

    tools_dict = None
    if tools:
        tools_dict = [tool.model_dump() for tool in tools]

    input_ids = tokenizer.apply_chat_template(
        formatted_messages, tools=tools_dict, tokenize=True, add_generation_prompt=True
    )

    if max_length and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    loss_mask = [0] * len(input_ids)

    current_pos = 0
    for msg in formatted_messages:
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        token_length = len(tokens)

        if msg["role"] == "assistant":
            approx_start = current_pos
            approx_end = approx_start + token_length

            for i in range(approx_start, min(approx_end, len(loss_mask))):
                loss_mask[i] = 1

        current_pos += token_length + 1

    return input_ids, loss_mask


def ids_to_messages(
    input_ids: torch.Tensor,
    tokenizer,
    skip_special_tokens: bool = True,
    role: str = "assistant",
) -> List[Message]:
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        if isinstance(input_ids[0], list):
            input_ids = [
                [tid for tid in seq if tid != pad_token_id] for seq in input_ids
            ]
        else:
            input_ids = [tid for tid in input_ids if tid != pad_token_id]

    text = tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)

    message = Message(role=role, content=text)

    return [message]

