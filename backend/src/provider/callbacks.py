from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from modelling.structured_output import Stats, StatsPatch
from storage.user import fetch_stats

from typing import Optional, Any, Callable


class CustomCallback(AsyncCallbackHandler):
    _config: RunnableConfig
    _store: BaseStore
    last_run_stats: Optional[Stats]
    current_run_patch: StatsPatch
    _max_retries: int
    retry_count: int

    def reset_retry_counter(self) -> None:
        """
        Helper function that reset the retry count.

        Returns:
            None
        """
        self.retry_count = self._max_retries

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Direct instantiation is not allowed. Use CustomCallback.create(...) instead.")

    @classmethod
    async def create(cls, config: RunnableConfig, store: BaseStore) -> "CustomCallback":
        """
        Factory method to create a CustomCallback instance.

        Args:
            config (RunnableConfig): The configuration for the runnable, containing runtime options and metadata.
            store (BaseStore): The storage backend used for persisting or retrieving data.

        Returns:
            CustomCallback: An initialized CustomCallback instance.
        """
        self = super().__new__(cls)
        self._config = config
        self._store = store
        # runtime benchmarking
        self.last_run_stats = await fetch_stats(self._config, self._store)
        self.current_run_patch = StatsPatch(token_usage=0)
        # critic agent retry logic
        self._max_retries = 2
        self.retry_count = self._max_retries
        super(CustomCallback, self).__init__()
        return self

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None, **kwargs: Any):
        # patch running conversation
        # user statistics with usage
        message: AIMessage = response.generations[0][0].message # type: ignore
        total_tokens = message.response_metadata.get("prompt_eval_count", 0) + message.response_metadata.get("eval_count", 0)
        # we only sum the running usage
        # as this cumulated run-patch is
        # further accumulated into the
        # profile statistics
        patch: StatsPatch = StatsPatch(token_usage=total_tokens)
        current = self.current_run_patch
        if isinstance(current, StatsPatch):
            self.current_run_patch = current.reduce(patch)

        return await super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)
