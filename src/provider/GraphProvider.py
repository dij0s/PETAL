import os
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_ollama import OllamaEmbeddings

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.redis.aio import AsyncRedisStore

from typing import Annotated, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel

from agent.intent_router import intent_router
from agent.clarify_query import clarify_query
from agent.geocontext_retriever import geocontext_retriever
from agent.generate_answer import generate_answer
from modelling.structured_output import State

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

    @classmethod
    def build(cls, redis_conn_string: str) -> "GraphProvider":
        """Provides an instance of GraphProvider.

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
        graph_builder.add_node("generate_answer", generate_answer)

        def router_condition(state: State):
            try:
                if state.router is not None and state.router.needs_clarification:
                    return "clarification"
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

        graph_builder.add_edge(START, "intent_router")
        graph_builder.add_edge("intent_router", END)
        graph_builder.add_conditional_edges("intent_router", router_condition)
        graph_builder.add_conditional_edges("geocontext_retriever", geocontext_condition)
        # reaching the clarification node should
        # stop the flow too to then process
        # extra user-given context
        graph_builder.add_edge("clarification", END)
        graph_builder.add_edge("generate_answer", END)
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

    async def stream_graph_generator(self, thread_id: str, user_id: str, user_input: str, lang: str = "en", with_state: bool = False) -> AsyncGenerator[tuple[str, Any], None]:
        """
        Asynchronously generates a stream of graph outputs based on user input.

        Args:
            thread_id (str): The thread identifier for the conversation.
            user_id (str): The unique user identifier.
            user_input (str): The user's input message to process in the graph.
            lang (str): The language code for processing. Defaults to "en".
            with_state (bool): If True, also passes the current state to the callback function. Defaults to False.

        Yields:
            AsyncGenerator[tuple[str, Any], None]: A tuple containing the mode (e.g., "token", "custom") and the corresponding output chunk.
        """
        stream_mode = ["messages", "custom"] if not with_state else ["messages", "custom", "values"]
        try:
            if isinstance(self._graph, CompiledStateGraph):
                configuration: dict = {
                    "configurable": {
                        "thread_id": thread_id,
                        "user_id": user_id,
                    }
                }

                async for mode, chunk in self._graph.astream(
                    {"messages": [HumanMessage(user_input)], "lang": lang},
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
