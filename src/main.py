import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

from langgraph.checkpoint.memory import InMemorySaver

from agent.intent_router import router_llm_node

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

async def chatbot(state: State):
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("router", router_llm_node)

graph_builder.add_edge(START, "router")

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

configuration = {
    "configurable": {
        "thread_id": "1"
    }
}

async def stream_graph_updates(user_input: str):
    async for event in graph.astream({"messages": [{"role": "user", "content": user_input}]}, configuration):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


async def main():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            await stream_graph_updates(user_input)
        except Exception as e:
            print(f"Error {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())
