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

# define system prompt for enhanced
# formatting and data scheme validation
tool_call_prompt = PromptTemplate.from_template("""
    You have access to these different tools:
    {tools_list}

    In response to the user's input, you can select and execute any number of tools from the available set.
    They will retrieve for you the data needed to answer the user input.
    They DON'T HAVE ANY ARGUMENTS, you can call them STRAIGHT AWAY.

    User input: "{aggregated_query}"
""")

llm = ChatOllama(model="llama3.2:3b", temperature=0)

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
            writer({"type": "custom_message", "content": "Let's start the machine."})
            provider = GeoSessionProvider.get_or_create(router_state.location, 100, 0.3)
            GeoSessionProvider.get_or_create(router_state.location, 100, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 500, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 1000, 1.0)
            writer({"type": "custom_message", "content": "Ok, that's done."})

            # await and fill context
            # with SFSO number of
            # municipality
            # await provider.wait_until_sfso_ready()
            # context.municipality_sfso_number = provider.municipality_sfso_number

            # create or get the singleton
            # tools provider instance
            # and retrieve relevant tools
            writer({"type": "custom_message", "content": "Where's my toolbox, I need my tools."})
            toolbox: ToolProvider = await ToolProvider.acreate(router_state.location)
            tools = await toolbox.asearch(query=router_state.aggregated_query, k=4)
            writer({"type": "custom_message", "content": "I FOUND THEM!"})

            # prompt to select best available tools
            tools_bound_llm = llm.bind_tools(tools=tools)
            tools_description = '\n'.join([f"-{t.name}: {t.description}" for t in tools])
            prompt = tool_call_prompt.format(tools_list=tools_description, aggregated_query=router_state.aggregated_query)
            writer({"type": "custom_message", "content": "Are these the right tools ?"})
            response = await tools_bound_llm.ainvoke(prompt)
            writer({"type": "custom_message", "content": "OK, the user guide said to use those.."})

            # invoke chosen tools
            # and update context state
            if isinstance(response, AIMessage) and hasattr(response, "tool_calls"):
                # retrieve tools by name
                tools_to_invoke = [
                    toolbox.get(tool["name"])
                    for tool in response.tool_calls
                ]
                if not any([tool is None for tool in tools_to_invoke]):
                    tool_data = await _ainvoke_tools(tools_to_invoke)
                    geocontext.context = {**geocontext.context, **tool_data}

                    return {
                        **state.model_dump(),
                        "messages": state.messages + [AIMessage(content="Successfully retrieved data from tools...")],
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
