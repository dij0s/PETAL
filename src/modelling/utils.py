import pydantic
from langchain_core.utils.pydantic import IS_PYDANTIC_V1

from typing import TypeVar

from functools import reduce

if IS_PYDANTIC_V1:
    PydanticBaseModel = pydantic.BaseModel
else:
    from pydantic.v1 import BaseModel
    PydanticBaseModel = BaseModel

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)

def construct_clarification_prompt(pydantic_object: TBaseModel) -> str:
    """
    Evaluates the given pydantic_object and returns a prompt for the user about missing attributes.

    Args:
        pydantic_object (TBaseModel): The Pydantic model instance to evaluate.

    Returns:
        str: A prompt indicating which required attributes are missing, or a message that all are present.
    """

    return "Provided instructions are unclear. Please provide this information more explicitly : " + ', '.join(reduce(
        lambda res, e: [*res, e[0]],
        filter(lambda e: e[1] is None, pydantic_object.model_dump().items()),
        []
    )) + "."
