import pydantic
from langchain_core.utils.pydantic import IS_PYDANTIC_V1

from typing import TypeVar, Optional

from functools import reduce

if IS_PYDANTIC_V1:
    PydanticBaseModel = pydantic.BaseModel
else:
    from pydantic.v1 import BaseModel
    PydanticBaseModel = BaseModel

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)

def reduce_missing_attributes(pydantic_object: TBaseModel) -> Optional[str]:
    """
    Evaluates the given pydantic_object and returns a prompt argument for the user about missing attributes and their description.

    Args:
        pydantic_object (TBaseModel): The Pydantic model instance to evaluate.

    Returns:
        Optional[str]: A reduced string with the required attributes which are missing or else None.
    """
    reduced_attributes = reduce(
        lambda res, e: [*res, f"- {e[0]}: {pydantic_object.model_fields[e[0]].description}"],
        filter(lambda e: e[1] is None, pydantic_object.model_dump().items()),
        []
    )

    if len(reduced_attributes) == 0:
        return None
    else:
        return '\n'.join(reduced_attributes)
