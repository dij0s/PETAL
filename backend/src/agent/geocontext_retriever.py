import re
import asyncio

from typing import Optional, Any

from functools import reduce

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools.structured import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter

from provider.ModelProvider import ModelProvider
from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider
from modelling.structured_output import RouterOutput, GeoContextOutput

llm_processing = (
    ModelProvider
        .from_env_variable(
            env_variable="OLLAMA_MODEL_LLM_PROCESSING",
            temperature=0,
            defaults="qwen3:1.7b",
            extract_reasoning=True
        )
)

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

system_prompt_processing = PromptTemplate.from_template("""
    You are a text processor. Your job is to scale energy numbers in the following documents.

    Instructions:
    - For each document, find energy-related numbers.
    - Multiply ONLY those numbers by {scaling_factor} and replace them in the text, rounded to 1 decimal place.
    - DO NOT scale percentages, dates, or any other numbers.
    - DO NOT add any explanations, notes, or comments.
    - DO NOT change any other part of the text.
    - Return the processed documents, separated by <doc>.

    Input documents:
    {constraints}

    Output:
    Return the same documents, in the same order, separated by <doc>. Only the relevant energy numbers should be changed.
    """)

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
            # start the instantiation of
            # the different GeoSession
            # for said location to reduce
            # latency when they are used
            # in the tools themselves
            writer({"type": "log", "content": "Let's start the machine."})
            provider = GeoSessionProvider.get_or_create(router_state.location, 100, 1.0, with_residents_count=True)
            GeoSessionProvider.get_or_create(router_state.location, 100, 0.3)
            GeoSessionProvider.get_or_create(router_state.location, 500, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 1000, 1.0)
            writer({"type": "log", "content": "Ok, that's done."})

            # retrieve relevant tools
            # and process constraints
            # for location-aware data
            writer({"type": "info", "content": "Retrieving tools and effective guidelines..."})
            shall_bypass_constraints: bool = router_state.intent == "factual"
            toolbox: ToolProvider = await ToolProvider.acreate(router_state.location)
            (tools, are_tools_uniform), constraints = await toolbox.asearch(query=router_state.aggregated_query, max_n_tools=6, k_tools=10, bypass_constraints=shall_bypass_constraints)
            writer({"type": "log", "content": "I FOUND THEM!"})
            # filter out tools whose
            # data we already have
            tools = [tool for tool in tools if tool.name not in geocontext.context_tools.keys()]

            # invoke necessary tools
            # and process constraints
            # concurrently if needed
            async def temp():
                return constraints

            writer({"type": "info", "content": "Fetching data from retrieved tools and processing guidelines.."})
            tool_data, processed_constraints = await asyncio.gather(
                _invoke_tools(tools, are_tools_uniform, router_state),
                _process_constraints(constraints, provider, shall_bypass_constraints)
                # temp()
            )
            # update context with
            # retrieved constraints
            # overwrite only as query
            # dependent
            geocontext.context_tools = {**geocontext.context_tools, **tool_data}
            geocontext.context_constraints = processed_constraints
            return {
                **state.model_dump(),
                "messages": state.messages + [AIMessage(content="Successfully retrieved data.")],
                "geocontext": geocontext,
            }

            return state
        else:
            # inquire extra clarification
            router_state.needs_clarification = True
            return {
                **state.model_dump(),
                "messages": state.messages,
                "router": router_state,
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

async def _process_constraints(constraints: list[tuple[str, str]], provider: GeoSessionProvider, bypass_constraints: bool = False) -> list[tuple[str, str]]:
    """
    Processes a list of constraints asynchronously.

    The state-wide constraints are processed for location-aware context.

    Args:
        constraints (list[tuple[str, str]]): A list of constraints tuple.
        provider (GeoSessionProvider): The provider for the given municipality.
        bypass_constraints (bool): If True, bypasses the constraints processing. Defaults to False.

    Returns:
        list[tuple[str, str]]: The list of location-aware constraints chunks and their source.
    """
    if (len(constraints) == 0) or bypass_constraints:
        return []

    # hardcoded population number
    # of canton as of now
    canton_population = 365844
    await provider.wait_until_residents_count_ready()
    SCALING_FACTOR = min(provider.residents_count / canton_population, 1)
    # retrieve documents
    constraints_chunks, constraints_sources = reduce(
        lambda res, c: ([*res[0], c[0]], [*res[1], c[1]]),
        constraints,
        ([], [])
    )

    prompt = [SystemMessage(content=system_prompt_processing.format(
        scaling_factor=SCALING_FACTOR,
        constraints="\n".join(f"<doc>{chunk}</doc>" for chunk in constraints_chunks),
    ))]
    # prompt the llm for the scaled
    # constraints specific for the
    # location
    response = await llm_processing.ainvoke(prompt)
    # extract the documents
    # from the response and
    # return original ones
    # on fallback
    try:
        document_pattern = re.compile(r"<doc>(.*?)</doc>", re.DOTALL)
        processed_constraints = [doc.strip() for doc in document_pattern.findall(response.content)] # type: ignore
        return reduce(
            lambda res, cs: [*res, (cs[0], cs[1])],
            zip(processed_constraints, constraints_sources),
            []
        )
    except:
        return constraints

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
