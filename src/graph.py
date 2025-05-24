import asyncio

from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage, BaseMessageChunk

from langgraph.checkpoint.memory import InMemorySaver

from typing import Annotated, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel

from modelling.structured_output import RouterOutput, GeoContextOutput

from agent.intent_router import intent_router
from agent.clarify_query import clarify_query
from agent.geocontext_retriever import geocontext_retriever

# overall state of the graph
class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    router: Optional[RouterOutput] = None
    geocontext: Optional[GeoContextOutput] = None

graph_builder = StateGraph(State)

graph_builder.add_node("intent_router", intent_router)
graph_builder.add_node("clarification", clarify_query)
graph_builder.add_node("geocontext_retriever", geocontext_retriever)

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
            return END
    except Exception as e:
        print(f"Error: {e}")

graph_builder.add_edge(START, "intent_router")
graph_builder.add_conditional_edges("intent_router", route_condition)
graph_builder.add_conditional_edges("geocontext_retriever", geocontext_condition)
# reaching the clarification node should
# stop the flow too to then process
# extra user-given context
graph_builder.add_edge("clarification", END)

# temporary short-term memory saver
# for conversation-like experience
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

configuration: RunnableConfig = {
    "configurable": {
        "thread_id": "1"
    }
}

async def stream_graph_generator(user_input: str) -> AsyncGenerator[str, None]:
    """Yield tokens one by one as strings for streaming."""
    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(user_input)]},
        config=configuration,
        stream_mode=["updates", "custom"]
    ):
        print(mode, chunk)
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            yield str(chunk.content)

async def stream_graph_updates(user_input: str, f: Callable[[str], None]):
    """Custom wrapper for tokens generator to print in CLI."""
    async for token in stream_graph_generator(user_input):
        f(token)
