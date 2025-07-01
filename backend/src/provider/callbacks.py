from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from modelling.structured_output import Stats, StatsPatch
from storage.user import fetch_stats, update_stats

from typing import Optional, Any

class CustomCallback(AsyncCallbackHandler):
    def __init__(self, config: RunnableConfig, store: BaseStore) -> None:
        """
        Custom callback that handles custom events and end of LLM generation.
        These implementations patch the current user's statistics with token usage and tool usage.

        Args:
            config (RunnableConfig): The configuration for the runnable, containing runtime options and metadata.
            store (BaseStore): The storage backend used for persisting or retrieving data.

        Returns:
            None
        """
        self._config = config
        self._store = store
        self.last_run_state: Optional[Stats] = None
        super().__init__()

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None, **kwargs: Any):
        # patch user statistics
        # with token usage
        message: AIMessage = response.generations[0][0].message # type: ignore
        total_tokens = message.response_metadata.get("prompt_eval_count", 0) + message.response_metadata.get("eval_count", 0)

        patch: StatsPatch = StatsPatch(token_usage=total_tokens)
        await update_stats(self._config, self._store, patch)

        return await super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)

    async def on_custom_event(self, name: str, data: Any, *, run_id: UUID, tags: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None, **kwargs: Any):
        # patch user statistics
        # with tool usage
        if (name == "tool_calls") and isinstance(data, int):
            patch: StatsPatch = StatsPatch(tool_usage=data)
            await update_stats(self._config, self._store, patch)
        # reset current run
        # memoized statistics
        elif (name == "memoize") and isinstance(data, Stats):
            self.last_run_state = data

        return await super().on_custom_event(name, data, run_id=run_id, tags=tags, metadata=metadata, **kwargs)
