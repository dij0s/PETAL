import asyncio

from typing import Optional, Any
from pydantic import BaseModel, Field, ValidationError

from functools import reduce

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.func import task
from langgraph.config import get_stream_writer

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from modelling.structured_output import RouterOutput, GeoContextOutput
from modelling.utils import reduce_missing_attributes

from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider

async def geocontext_retriever(state):
    """
    Retrieves relevant geographical and contextual data based on the user query.

    This function:
      - Extracts the last human message from the state.
      - Formats a routing prompt using a predefined template.
      - Invokes a language model to classify and summarize the user's query.
      - Retrieves relevant geographic and contextual data based on the classified intent.
      - Augments the conversation state with this retrieved contextual information.
      - Handles retrieval errors and determines if additional data sources are needed.

    Args:
        state: The current conversation state to which we add the retrieved geo-context.

    Returns:
        dict: The updated conversation state with.
    """
    writer = get_stream_writer()

    last_human_message = next(msg.content for msg in state.messages if isinstance(msg, HumanMessage))

    geocontext: Optional[GeoContextOutput] = state.geocontext
    if geocontext is None:
        geocontext = GeoContextOutput()

    router_state: RouterOutput = state.router
    try:
        # instantiate potentially needed
        # geometry sessions and schemas
        # based on router location
        # also check that aggregated query
        # is set for type safety but, logically
        # speaking, it is set if we are inside
        # the current node
        if router_state.location is not None and router_state.aggregated_query is not None:
            # write custom event
            writer({"type": "log", "content": "Let's start the machine."})
            provider = GeoSessionProvider.get_or_create(router_state.location, 100, 0.3)
            GeoSessionProvider.get_or_create(router_state.location, 100, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 500, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 1000, 1.0)
            writer({"type": "log", "content": "Ok, that's done."})

            # await and fill context
            # with SFSO number of
            # municipality
            # await provider.wait_until_sfso_ready()
            # context.municipality_sfso_number = provider.municipality_sfso_number

            # create or get the singleton
            # tools provider instance
            # and retrieve relevant tools
            writer({"type": "info", "content": "Retrieving tools..."})
            toolbox: ToolProvider = await ToolProvider.acreate(router_state.location)
            tools, constraints = await toolbox.asearch(query=router_state.aggregated_query, max_n=3, k=5)
            print(constraints)
            writer({"type": "log", "content": "I FOUND THEM!"})
            # filter out tools whose
            # data we already have
            tools = [tool for tool in tools if tool.name not in geocontext.context.keys()]
            # invoke chosen tools
            # and update context state
            if len(tools) > 0:
                writer({"type": "info", "content": "Fetching data from retrieved tools..."})
                tool_data = await _ainvoke_tools(tools)
                geocontext.context = {**geocontext.context, **tool_data}
            else:
                # needed data is already retrieved
                writer({"type": "info", "content": "We alread have them!"})

            return {
                **state.model_dump(),
                "messages": state.messages + [AIMessage(content="Successfully retrieved data.")],
                "geocontext": geocontext
            }

            return state
        else:
            # inquire extra clarification
            router_state.needs_clarification = True
            return {
                **state.model_dump(),
                "messages": state.messages,
                "router": router_state
            }
    except Exception as e:
        print(f"Exception: {e}")
        return state

async def _ainvoke_tools(tools: list[StructuredTool]) -> dict[str, Any]:
    """Helper function that invokes a batch of tools asynchronously and returns the result."""

    data: list[dict[str, Any]] = await asyncio.gather(
        *(tool.coroutine() for tool in tools if tool.coroutine is not None)
    )
    # reduce partial results
    # to single dictionnary
    return {
        k: v
        for d in data
        for k, v in d.items()
    }
