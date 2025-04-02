from .data_model import OpenAIFunctionToolSchema, OpenAIFunctionParametersSchema, OpenAIFunctionParsedSchema


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
    
    def create(self) -> None:
        pass
    
    def execute(self, parameters: OpenAIFunctionParsedSchema) -> None:
        pass
    
    def calc_reward(self) -> float:
        return 0.0
    
    def release(self) -> None:
        pass
    
    