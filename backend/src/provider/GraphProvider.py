import os
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_ollama import OllamaEmbeddings

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.redis.aio import AsyncRedisStore

from typing import AsyncGenerator, Optional, Callable, Any

from agent.intent_router import intent_router
from agent.clarify_query import clarify_query
from agent.geocontext_retriever import geocontext_retriever
from agent.guidelines_retriever import guidelines_retriever
from agent.generate_answer import generate_answer
from agent.critic_answer import critic_answer
from provider.callbacks import CustomCallback
from modelling.structured_output import State, CriticOutput, Stats, StatsPatch
from storage.user import fetch_stats

class GraphProvider:
    """
    Manages the lifecycle and configuration of the
    LangGraph StateGraph for conversational flows.

    Follows the context provider pattern for consumers.
    """

    def __init__(self, redis_conn_string: str) -> None:
        self._redis_conn_string: str = redis_conn_string
        # temporary short-term memory saver
        # for conversation-like experience
        self._checkpointer: InMemorySaver = InMemorySaver()
        # long-term memory saver
        # for user memories
        self._store: Optional[AsyncRedisStore] = None
        self._graph: Optional[CompiledStateGraph] = None
        # runtime benchmarking
        self.last_run_stats: Optional[Stats] = None
        self.current_run_patch: StatsPatch = StatsPatch(token_usage=0)
        # critic agent retry logic
        self._max_retries: int = 2
        self._retry_count: int = self._max_retries

    def _get_last_run_stats(self) -> Optional[Stats]:
        """
        Returns the statistics from the last run.
        """
        return self.last_run_stats

    def _set_last_run_stats(self, stats: Stats) -> None:
        """
        Sets the statistics for the last run.
        """
        self.last_run_stats = stats

    def _get_current_run_patch(self) -> StatsPatch:
        """
        Returns the current run patch.
        """
        return self.current_run_patch

    def _set_current_run_patch(self, patch: StatsPatch) -> None:
        """
        Sets the current run patch.
        """
        self.current_run_patch = patch

    def _get_retry_count(self) -> int:
        """
        Helper function that returns the current retry count and decrements it.

        Returns:
            int: The current retry count.
        """
        if self._retry_count < 0:
            self._retry_count = self._max_retries

        current = self._retry_count
        self._retry_count -= 1
        return current

    def _reset_retry_count(self) -> None:
        """
        Helper function that reset the retry count.

        Returns:
            None
        """
        self._retry_count = self._max_retries

    @classmethod
    def build(cls, redis_conn_string: str) -> "GraphProvider":
        """
        Provides an instance of GraphProvider.

        Args:
            redis_conn_string (str): The Redis long-term memory store connection string.

        Returns:
            GraphProvider: An instance of the GraphProvider class.
        """
        return cls(redis_conn_string)

    async def __aenter__(self) -> "GraphProvider":
        EMBEDDING_MODEL = os.getenv("OLLAMA_MODEL_EMBEDDING", "nomic-embed-text:v1.5")
        EMBEDDING_MODEL_DIMS = os.getenv("OLLAMA_MODEL_EMBEDDING_DIMS", "768")
        embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)

        index = {
            "dims": EMBEDDING_MODEL_DIMS,
            "embed": embedder,
            "fields": ["memory", "context"]
        }
        # instantiate store without
        # using API builtin provider
        # pattern to handle lifecycle
        # manually
        self._store = await AsyncRedisStore(redis_url=self._redis_conn_string, index=index).__aenter__() # type: ignore
        # graph definition
        graph_builder = StateGraph(State)
        graph_builder.add_node("intent_router", intent_router)
        graph_builder.add_node("clarification", clarify_query)
        graph_builder.add_node("geocontext_retriever", geocontext_retriever)
        graph_builder.add_node("guidelines_retriever", guidelines_retriever)
        graph_builder.add_node("generate_answer", generate_answer)
        graph_builder.add_node("critic_answer", critic_answer)

        def router_condition(state: State):
            try:
                if state.router is not None:
                    if state.router.needs_clarification or (state.router.location is None and state.router.aggregated_query is None):
                        return "clarification"
                    elif state.router.conversation_type == "correction_request":
                        return "geocontext_retriever"
                    elif state.router.intent == "actionable":
                        return "geocontext_retriever", "guidelines_retriever"
                    else:
                        return "geocontext_retriever"

            except Exception as e:
                print(f"Error: {e}")

        def geocontext_condition(state: State):
            try:
                if state.router is not None and state.router.needs_clarification:
                    return "clarification"
                else:
                    return "generate_answer"
            except Exception as e:
                print(f"Error: {e}")

        def guidelines_condition(state: State):
            try:
                if state.router is not None and state.router.needs_clarification:
                    return "clarification"
                else:
                    return "generate_answer"
            except Exception as e:
                print(f"Error: {e}")

        def critic_condition(state: CriticOutput):
            if state.retry:
                return "intent_router"
            else:
                return END

        graph_builder.add_edge(START, "intent_router")
        graph_builder.add_conditional_edges("intent_router", router_condition)
        graph_builder.add_conditional_edges("geocontext_retriever", geocontext_condition)
        graph_builder.add_conditional_edges("guidelines_retriever", guidelines_condition)
        graph_builder.add_edge("generate_answer", "critic_answer")
        # reaching the clarification node should
        # stop the flow too to then process
        # extra user-given context
        graph_builder.add_conditional_edges("critic_answer", critic_condition)
        graph_builder.add_edge("clarification", END)
        # compile graph and define
        # runtime configuration
        self._graph = graph_builder.compile(checkpointer=self._checkpointer, store=self._store)

        return self

    async def __aexit__(self, exc_type, exc, tb):
        # properly handle lifecycle
        # of redis store from wrapper
        # context provider
        if self._store:
            await self._store.__aexit__(exc_type, exc, tb)

    async def stream_graph_generator(self, thread_id: str, user_id: str, user_input: str, lang: str = "en", with_state: bool = False, initial_state: Optional[State] = None) -> AsyncGenerator[tuple[str, Any], None]:
        """
        Asynchronously generates a stream of graph outputs based on user input.

        Args:
            thread_id (str): The thread identifier for the conversation.
            user_id (str): The unique user identifier.
            user_input (str): The user's input message to process in the graph.
            lang (str): The language code for processing. Defaults to "en".
            with_state (bool): If True, also passes the current state to the callback function. Defaults to False.
            initial_state (Optional[State]): Initial state for rehydrating checkpointed conversations.

        Yields:
            AsyncGenerator[tuple[str, Any], None]: A tuple containing the mode (e.g., "token", "custom") and the corresponding output chunk.
        """
        stream_mode = ["messages", "custom"] if not with_state else ["messages", "custom", "values"]
        try:
            if isinstance(self._graph, CompiledStateGraph):
                # runtime configuration
                configurable: dict =  {
                    "configurable": {
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "retry_handlers": (self._get_retry_count, self._reset_retry_count),
                        "last_run_handlers": (self._get_last_run_stats, self._set_last_run_stats),
                        "current_run_handlers": (self._get_current_run_patch, self._set_current_run_patch)
                    }
                }
                configuration: dict = {
                    **configurable,
                    "callbacks": [CustomCallback(configurable, self._store)] # type: ignore
                }
                # retrieve user statistics
                # from database if not done
                # hence on very first run
                if self.last_run_stats is None:
                    self.last_run_stats = await fetch_stats(configuration, self._store) # type: ignore
                # prepare the input state
                # depending on the initial
                # state for rehydratation
                input_state = {}
                if initial_state:
                    input_state = initial_state.model_copy()
                    input_state.messages = [HumanMessage(user_input)]
                    input_state.lang = lang
                else:
                    input_state = {"messages": [HumanMessage(user_input)], "lang": lang}

                async for mode, chunk in self._graph.astream(
                    input_state,
                    config=configuration, # type: ignore
                    stream_mode=stream_mode # type: ignore
                ):
                    if mode == "messages":
                        token, metadata = chunk
                        if (
                            isinstance(token, AIMessageChunk)
                            and isinstance(metadata, dict)
                            and metadata.get("langgraph_node") in ["clarification", "generate_answer"]
                        ):
                            yield "token", token.content
                    elif mode == "custom":
                        if isinstance(chunk, dict) and chunk.get("type") != "log":
                            yield chunk.get("type"), json.dumps(chunk) # type: ignore
                    else:
                        yield mode, chunk
            else:
                raise Exception("GraphProvider must be instantiated before using it.")
        except Exception as e:
            print(f"Exception: {e}")

    async def stream_graph_updates(self, thread_id: str, user_id: str, user_input: str, f: Callable[[str, Any], None], with_state: bool = False):
        """
        Asynchronously streams graph updates based on user input and applies a callback function.
        Implemented for CLI use.

        Args:
            thread_id (str): The thread identifier for the conversation.
            user_id (str): The unique user identifier.
            user_input (str): The user's input message to process in the graph.
            f (Callable[[str, Any], None]): A callback function that processes each output chunk's mode and content from the graph.
            with_state (bool): If True, also passes the current state to the callback function. Defaults to False.

        Yields:
            None: This method does not yield but calls the callback function for each output.
        """
        async for mode, chunk in self.stream_graph_generator(
            thread_id=thread_id,
            user_id=user_id,
            user_input=user_input,
            with_state=with_state
        ):
            f(mode, chunk) # type: ignore
