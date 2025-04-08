from pydantic import BaseModel


class OpenAIFunctionPropertySchema(BaseModel):
    """The schema of a parameter in OpenAI format."""
    type: str
    description: str | None = None
    enum: list[str] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    """The schema of parameters in OpenAI format."""
    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    """The schema of a function in OpenAI format."""
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema


class OpenAIFunctionToolSchema(BaseModel):
    """The schema of a tool in OpenAI format."""
    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""
    name: str
    arguments: dict[str, str]
