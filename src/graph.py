import asyncio

from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

from typing import Annotated, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel

from agent import generate_answer
from modelling.structured_output import RouterOutput, GeoContextOutput

from agent.intent_router import intent_router
from agent.clarify_query import clarify_query
from agent.geocontext_retriever import geocontext_retriever
from agent.generate_answer import generate_answer

# overall state of the graph
class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    router: Optional[RouterOutput] = None
    geocontext: Optional[GeoContextOutput] = None

graph_builder = StateGraph(State)

graph_builder.add_node("intent_router", intent_router)
graph_builder.add_node("clarification", clarify_query)
graph_builder.add_node("geocontext_retriever", geocontext_retriever)
graph_builder.add_node("generate_answer", generate_answer)

def route_condition(state: State):
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
graph_builder.add_conditional_edges("intent_router", route_condition)
graph_builder.add_conditional_edges("geocontext_retriever", geocontext_condition)
# reaching the clarification node should
# stop the flow too to then process
# extra user-given context
graph_builder.add_edge("clarification", END)
graph_builder.add_edge("generate_answer", END)

# temporary short-term memory saver
# for conversation-like experience
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

configuration: RunnableConfig = {
    "configurable": {
        "thread_id": "1"
    }
}

async def stream_graph_generator(user_input: str) -> AsyncGenerator[tuple[str, Any], None]:
    """Yield tokens one by one as strings for streaming."""
    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(user_input)]},
        config=configuration,
        stream_mode=["messages", "custom"]
    ):
        # only yield individual tokens
        # coming from last node in the
        # graph
        if mode == "messages":
            token, metadata = chunk
            if (
                isinstance(token, AIMessageChunk)
                and isinstance(metadata, dict)
                and metadata.get("langgraph_node") in ["clarification", "generate_answer"]
            ):
                print(mode, token)
                yield "token", token.content
        elif mode == "custom":
            if (
                isinstance(chunk, dict)
                and chunk.get("type") != "log"
            ):
                yield chunk.get("type"), chunk

async def stream_graph_updates(user_input: str, f: Callable[[Any], None]):
    """Custom wrapper for tokens generator to print in CLI."""
    async for mode, chunk in stream_graph_generator(user_input):
        f(chunk)
