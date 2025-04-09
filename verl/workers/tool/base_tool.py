from typing import Optional
from uuid import uuid4
from .data_model import OpenAIFunctionToolSchema

class BaseTool(object):
    """Base class for tools.

    A tool should support the following methods:
    
    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.name = tool_schema.function.name
        self.tool_schema = tool_schema

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    def create(self, instance_id: Optional[str] = None) -> str:
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id
    
    def execute(self, instance_id: str, parameters: str) -> str:
        return "Updated the tool state."
    
    def calc_reward(self, instance_id: str) -> float:
        return 0.0
    
    def release(self, instance_id: str) -> None:
        pass
    
    