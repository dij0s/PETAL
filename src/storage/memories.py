import asyncio
import uuid
import time

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore

from modelling.structured_output import Memory

async def fetch_memories(config: RunnableConfig, store: BaseStore, query: str, limit: int = 3) -> list[Memory]:
    """
    Fetches the user's memories from the long-term memory store.

    This function retrieves all memories associated with the user specified in the
    provided user_id from the given memory store.

    Args:
        config: The configuration for the runnable.
        store (BaseStore): The long-term memory store to fetch memories from.
        query (str): The query string to search for relevant memories.
        limit (int, optional): The maximum number of memories to fetch. Defaults to 3.

    Returns:
        list[Memory]: A list of Memory items.
    """
    try:
        user_id = config["configurable"].get("user_id") # type: ignore
        if user_id is None:
            raise Exception("Could not retrieve user_id from runtime configuration.")

        namespace = ("memories", user_id)
        return [
            Memory(**item.value)
            for item in await store.asearch(namespace, query=query, limit=limit)
        ]
    except Exception as e:
        print(f"Exception: {e}")
        return []

async def update_memories(config: RunnableConfig, store: BaseStore, last_human_message: str, previous_human_message: str) -> None:
    """
    Updates the user's memories in the long-term memory store.

    This function:
        - Extracts relevant information from the current conversation state.
        - Updates or adds user-related memories in the provided long-term memory store.
        - Ensures that the latest user context is persisted for future retrieval.

    Args:
        config: The configuration for the runnable.
        store: The long-term memory store.
        last_human_message: The most recent message from the user.
        previous_human_message: The message from the user prior to the most recent one.

    Returns:
        dict: The updated conversation state.
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
                }
            )
        except Exception as e:
            print(f"Exception: {e}")

    # start update task
    # in the background
    asyncio.create_task(helper())
    return
