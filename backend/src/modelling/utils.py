import pydantic
from langchain_core.utils.pydantic import IS_PYDANTIC_V1

from .structured_output import Stats, StatsPatch

from functools import reduce
import time
from typing import TypeVar, Optional

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

def welford_single_pass_accumulator(old: Optional[Stats], patch: StatsPatch) -> Stats:
    """
    Accumulates statistics using Welford's single-pass algorithm.

    Args:
        old (Optional[Stats]): The previous accumulated statistics.
        new (StatsPatch): The statistics patch to incorporate.

    Returns:
        Stats: The updated statistics after incorporating the new values.
    """
    # initialize stats or
    # update them if they
    # exist for given user
    if old is None:
        return Stats(
            token_usage_mean = patch.token_usage or 0,
            token_usage_M2 = 0,
            chat_calls_count = 1 if patch.token_usage is not None else 0,
            timestamp = time.time()
        )
    else:
        # update record with current user
        # stats as per Welford's online
        # algorithm which provides a numerically
        # stable algorithm with a recurrence
        # relation to help enable us to compute
        # the variance and sampled variance in
        # a single pass
        if patch.token_usage is not None:
            new_chat_calls_count = old.chat_calls_count + 1
            delta = patch.token_usage - old.token_usage_mean
            new_token_usage_mean = old.token_usage_mean + (delta / new_chat_calls_count)
            new_token_usage_M2 = old.token_usage_M2 + delta * (patch.token_usage - new_token_usage_mean)

            return Stats(
                **{
                    "token_usage_mean": new_token_usage_mean,
                    "token_usage_M2": new_token_usage_M2,
                    "chat_calls_count": new_chat_calls_count,
                    "timestamp": time.time()
                }
            )
        else:
            return old

def bin(old: Optional[Stats], new: Stats) -> int:
    """
    Bins the given stats into a single integer value representing the overall "greenness" score.

    -1  ->  bad
     0  ->  ok
     1  ->  good

    Args:
        old (Optional[Stats]): The previous stats.
        new (Stats): The new stats to bin.

    Returns:
        int: The binned value.
    """
    if (old is None) or (old.chat_calls_count < 2):
        return 0

    zscore_tokens = 0
    # std computed per Welford definition
    if old.chat_calls_count >= 2:
        std = old.std()
        if std > 0:
            zscore_tokens = (new.token_usage_mean - old.token_usage_mean) / std
        else:
            zscore_tokens = 0

    # zscore value being the
    # distance between the raw
    # score and the pop. mean
    # in units of standard dev.
    # implies we can evaluate
    # against a simple scalar
    if zscore_tokens < -1:
        return 1
    elif zscore_tokens > 1:
        return -1
    else:
        return 0
