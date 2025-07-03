import asyncio
import uuid
import time
import numpy as np

from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore

from sentence_transformers import CrossEncoder

from typing import Optional
from modelling.structured_output import Memory, Stats, StatsPatch
from modelling.utils import welford_single_pass_accumulator

_reranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

async def update_memories(config: RunnableConfig, store: BaseStore, last_human_message: str, previous_human_message: str) -> None:
    """
    Updates the user's memories in the long-term memory store.

    This function:
        - Extracts relevant information from the current conversation state.
        - Updates or adds user-related memories in the provided long-term memory store.
        - Ensures that the latest user context is persisted for future retrieval.

    Args:
        config (RunnableConfig): The configuration for the runnable.
        store (BaseStore): The long-term memory store.
        last_human_message (str): The most recent message from the user.
        previous_human_message (str): The message from the user prior to the most recent one.

    Returns:
        None
    """

    async def helper() -> None:
        """
        This function actually updates the user's memories.
        """
        try:
            user_id = config["configurable"].get("user_id") # type: ignore
            if user_id is None:
                raise Exception("Could not retrieve user_id from runtime configuration.")

            namespace = ("memories", user_id)

            # the previous message human
            # message is considered as extra
            # context as when user says "A",
            # he actually means "A" | "B"
            await store.aput(
                namespace,
                str(uuid.uuid4()),
                {
                    "memory": last_human_message,
                    "context": previous_human_message,
                    "timestamp": time.time()
                },
            )
        except Exception as e:
            print(f"Exception: {e}")

    # start update task
    # in the background
    # to avoid blocking
    # the caller
    asyncio.create_task(helper())
    return

async def fetch_memories(config: RunnableConfig, store: BaseStore, query: str) -> list[Memory]:
    """
    Fetches the user's memories from the long-term memory store.

    This function retrieves all memories associated with the user specified in the
    provided user_id from the given memory store.

    Args:
        config (RunnableConfig): The configuration for the runnable.
        store (BaseStore): The long-term memory store to fetch memories from.
        query (str): The query string to search for relevant memories.

    Returns:
        list[Memory]: A list of Memory items.
    """
    try:
        user_id = config["configurable"].get("user_id") # type: ignore
        if user_id is None:
            raise Exception("Could not retrieve user_id from runtime configuration.")

        namespace = ("memories", user_id)
        # retrieve memories and apply
        # rerank for better relevance
        # assessment
        memories = [
            Memory(**item.value)
            for item in await store.asearch(namespace, query=query, limit=5)
        ]
        if len(memories) == 0:
            return []

        pairs = [(query, item.context) for item in memories]
        # no need to batch as at most
        # 5 memories are retrieved
        logits = _reranking_model.predict(pairs)
        # apply softmax normalization
        # and threshold before selecting
        # the most relevant ones
        exp_logits = np.exp(logits - np.max(logits))
        scores = exp_logits / np.sum(exp_logits)
        # threshold the relevant items
        # using the mean score (1 being the sum p.d.)
        threshold = 1 / len(memories)

        top_indices = [
            index
            for index, score in enumerate(scores)
            if score > threshold
        ]
        return [memories[index] for index in top_indices]
    except Exception as e:
        print(f"Exception: {e}")
        return []

async def update_stats(config: RunnableConfig, store: BaseStore, old: Optional[Stats], patch: StatsPatch) -> Optional[Stats]:
    """
    Updates the user's statistics in the long-term memory store.

    Args:
        config (RunnableConfig): The configuration for the runnable.
        store (BaseStore): The long-term memory store.
        old (Optional[Stats]): The current user statistics.
        patch (StatsPatch): The patch to apply to the user's statistics.

    Returns:
        Optional[Stats]: The updated user statistics if successful, None otherwise.
    """

    try:
        user_id = config["configurable"].get("user_id") # type: ignore
        if user_id is None:
            raise Exception("Could not retrieve user_id from runtime configuration.")

        namespace = ("stats",)

        new = welford_single_pass_accumulator(old, patch)
        # documents key is user_id
        await store.aput(
            namespace,
            user_id,
            new.model_dump()
        )
        return new
    except Exception as e:
        print(f"Exception: {e}")

    return None

async def fetch_stats(config: RunnableConfig, store: BaseStore) -> Optional[Stats]:
    """
    Fetches the user's statistics record from the long-term memory store.

    Args:
        config (RunnableConfig): The configuration for the runnable.
        store (BaseStore): The long-term memory store to fetch memories from.

    Returns:
        Optional[Stats]: A Stats record containing the user's statistics if exists, else None.
    """
    try:
        user_id = config["configurable"].get("user_id") # type: ignore
        if user_id is None:
            raise Exception("Could not retrieve user_id from runtime configuration.")

        namespace = ("stats",)

        item = await store.aget(namespace, user_id)
        if item is not None:
            return Stats(**item.value)
    except Exception as e:
        print(f"Exception: {e}")

    return None
