import pydantic
from langchain_core.utils.pydantic import IS_PYDANTIC_V1

from typing import TypeVar, Optional
from modelling.structured_output import Stats

from math import ceil, sqrt
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

def bin(old: Stats, new: Stats) -> int:
    """
    Bins the given stats into a single integer value representing the overall "greenness" score.

    -1  ->  bad
     0  ->  ok
     1  ->  good

    Args:
        old (Stats): The previous stats.
        new (Stats): The new stats to bin.

    Returns:
        int: The binned value.
    """
    if (old.chat_calls_count < 2) and (old.tool_calls_count < 2):
        return 1

    score_tokens = 0
    score_tools = 0
    # std computed per Welford definition
    # n denominator -> variance which
    # implies we don't compare against
    # sampled variance but assume it to
    # be the distribution's one
    if old.chat_calls_count >= 2:
        std = sqrt(old.token_usage_M2 / old.chat_calls_count)
        if std > 0:
            zscore = (new.token_usage_mean - old.token_usage_mean) / std
            # zscore value being the
            # distance between the raw
            # score and the pop. mean
            # in units of standard dev.
            # implies we can evaluate
            # against a simple scalar
            if zscore < -0.5:
                score_tokens = 1
            elif zscore > 1.0:
                score_tokens = -1
            else:
                score_tokens = 0
        else:
            score_tokens = 1

    if old.tool_calls_count >= 2:
        std = sqrt(old.tool_usage_M2 / old.tool_calls_count)
        if std > 0:
            zscore = (new.tool_usage_mean - old.tool_usage_mean) / std
            if zscore < -0.5:
                score_tools = 1
            elif zscore > 1.0:
                score_tools = -1
            else:
                score_tools = 0
        else:
            score_tools = 1

    mean_score = (score_tokens + score_tools) / 2
    return ceil(mean_score)
