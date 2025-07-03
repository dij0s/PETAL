from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from modelling.structured_output import StatsPatch

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
        # destructure graph-wide state
        # handlers into instance methods
        self.get_current_run_patch, self.set_current_run_patch = (
            self._config
                .get("configurable", {})
                .get("current_run_handlers")
        ) # type: ignore
        self.get_last_run_stats, self.set_last_run_stats = (
            self._config
                .get("configurable", {})
                .get("last_run_handlers")
        ) # type: ignore
        # retrieve retry handlers
        # from configuration
        self.get_retry_counter, self.reset_retry_counter = (
            config
                .get("configurable", {})
                .get("retry_handlers")
        ) # type: ignore
        super().__init__()

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
        if (self.get_current_run_patch is not None) and (self.set_current_run_patch is not None):
            current = self.get_current_run_patch()
            if isinstance(current, StatsPatch):
                self.set_current_run_patch(current.reduce(patch))

        return await super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)
