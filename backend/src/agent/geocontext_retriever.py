import re
import asyncio

from typing import Optional, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools.structured import StructuredTool
from langgraph.config import get_stream_writer

from provider.ModelProvider import ModelProvider
from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider
from modelling.structured_output import RouterOutput, GeoContextOutput

llm_tool_retrieval = (
    ModelProvider
        .from_env_variable(
            env_variable="OLLAMA_MODEL_LLM_TOOLS",
            temperature=0,
            defaults="qwen3:1.7b",
        )
)

system_prompt_tool_retrieval = PromptTemplate.from_template("""
    You are an energy planning expert and you are given the following task :

    In response to the user's input, you can select and execute any number of tools from the available set.
    They will retrieve for you the data needed to answer the user input.
    **IMPORTANT: The tools don't require any configuration.**

    **Please note that all tools allow you to retrieve data specific to "{location}"**

    ### User Request: "{user_request}"
    """)

async def geocontext_retriever(state):
    """
    Function for retrieving and augmenting the conversation state with relevant geographic data.

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
        if router_state.location is None or router_state.aggregated_query is None:
            raise ValueError("State is undefined")
        # start the instantiation of
        # the different GeoSession
        # for said location to reduce
        # latency when they are used
        # in the tools themselves
        writer({"type": "log", "content": "Let's start the machine."})
        GeoSessionProvider.get_or_create(router_state.location, 1000, 1.0)
        GeoSessionProvider.get_or_create(router_state.location, 500, 1.0)
        GeoSessionProvider.get_or_create(router_state.location, 100, 1.0)
        GeoSessionProvider.get_or_create(router_state.location, 100, 0.3)
        writer({"type": "log", "content": "Ok, that's done."})

        # retrieve relevant tools
        # for location-aware data
        writer({"type": "info", "content": "Retrieving tools..."})
        toolbox: ToolProvider = await ToolProvider.acreate(router_state.location)
        tools, are_tools_uniform = await toolbox.asearch_tools(query=router_state.aggregated_query, max_n=6, k=10)
        writer({"type": "log", "content": "I FOUND THEM!"})
        # filter out tools whose
        # data we already have
        tools = [tool for tool in tools if tool.name not in geocontext.context_tools.keys()]

        # invoke necessary tools
        writer({"type": "info", "content": "Fetching data from retrieved tools..."})
        try:
            tool_data = await _invoke_tools(tools, are_tools_uniform, router_state)
        except RuntimeError:
            print("Shit going downhill in geocontext retriever")
            # location is not a proper
            # municipalty, enquire more
            # clarification by unsetting
            # the non-valid location
            router_state.location = None
            router_state.needs_clarification = True
            return {
                "router": router_state
            }
        # update context
        geocontext.context_tools = {**geocontext.context_tools, **tool_data}
        return {
            "messages": [AIMessage(content="Successfully retrieved data.")],
            "geocontext": geocontext,
        }
    except Exception as e:
        print(f"Exception: {e}")
        return state

async def _invoke_tools(tools: list[StructuredTool], are_tools_uniform: bool, router_state: RouterOutput) -> dict[str, Any]:
    """
    Invokes a list of StructuredTool objects asynchronously and returns their results as a dictionary.

    This function checks if there are tools to invoke. If the probability distribution of the tools is uniform,
    it prompts the language model to help select the most relevant tools for the user's query. It then fetches
    data from the selected tools asynchronously.

    Args:
        tools (list[StructuredTool]): The list of tools to be invoked.
        are_tools_uniform (bool): Indicates if the probability distribution of the tools is uniform.
        router_state (RouterOutput): The current router state containing location and query information.

    Returns:
        dict[str, Any]: A dictionary containing the results from the invoked tools. If no tools are invoked,
                        returns an empty dictionary.
    """
    if len(tools) > 0:
        # prompt the llm to better
        # choose the tools if the
        # retrieved tools underlying
        # probability distribution
        # is uniform
        if are_tools_uniform:
            tools_bound_llm = llm_tool_retrieval.bind_tools(tools)
            response = await tools_bound_llm.ainvoke([SystemMessage(content=system_prompt_tool_retrieval.format(
                location=router_state.location,
                user_request=router_state.aggregated_query
            ))])

            toolbox: ToolProvider = await ToolProvider.acreate(router_state.location) # type: ignore
            # if no tools were chosen
            # by the llm, default to
            # the larger distribution
            if hasattr(response, "tool_calls"):
                tools = [
                    toolbox.get(tool.get("name"))
                    for tool in response.tool_calls # type: ignore
                ]  # type: ignore

        return await _ainvoke_tools(tools)
    else:
        return {}

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
