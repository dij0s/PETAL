import os
import re
import asyncio

from typing import Optional, Any, Awaitable
from pydantic import BaseModel, Field, ValidationError

from functools import reduce, partial

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.func import task
from langgraph.config import get_stream_writer

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from modelling.structured_output import RouterOutput, GeoContextOutput, ConstraintsOutput
from modelling.utils import reduce_missing_attributes

from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider

MODEL = os.getenv("OLLAMA_MODEL_LLM_PROCESSING", "llama3.2:3b")
llm = ChatOllama(model=MODEL, temperature=0, extract_reasoning=True)

parser = PydanticStreamOutputParser(pydantic_object=ConstraintsOutput, diff=True)

processing_prompt = PromptTemplate.from_template("""
Scale energy numbers in these documents by multiplying by {scaling_factor}.

Documents:
{constraints}

Rules:
- Find numbers followed by GWh, MWh, kWh, MW, kW
- Multiply those numbers by {scaling_factor}
- Keep everything else exactly the same
- Return each document separately

Examples:
- "1,400 GWh" becomes "70.0 GWh"
- "2,790 GWh/a" becomes "139.5 GWh/a"

{format_description_llm}
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
            provider = GeoSessionProvider.get_or_create(router_state.location, 100, 0.3)
            GeoSessionProvider.get_or_create(router_state.location, 100, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 500, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 1000, 1.0)
            writer({"type": "log", "content": "Ok, that's done."})
            # retrieve relevant tools
            # and process constraints
            # for location-aware data
            writer({"type": "info", "content": "Retrieving tools and effective guidelines..."})
            toolbox: ToolProvider = await ToolProvider.acreate(router_state.location)
            tools, constraints = await toolbox.asearch(query=router_state.aggregated_query, max_n_tools=5, k_tools=10)
            writer({"type": "log", "content": "I FOUND THEM!"})
            # filter out tools whose
            # data we already have
            tools = [tool for tool in tools if tool.name not in geocontext.context_tools.keys()]
            async def _helper():
                if len(tools) > 0:
                    writer({"type": "info", "content": "Fetching data from retrieved tools..."})
                    return await _ainvoke_tools(tools)
                else:
                    # needed data is already retrieved
                    writer({"type": "info", "content": "We already have them!"})
                    return {}
            # invoke necessary tools
            # and process constraints
            # concurrently
            tool_data, processed_constraints = await asyncio.gather(
                _helper(),
                _process_constraints(constraints, provider)
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

async def _process_constraints(constraints: list[tuple[str, str]], provider: GeoSessionProvider) -> list[tuple[str, str]]:
    """
    Processes a list of constraints asynchronously.

    The state-wide constraints are processed for location-aware context.

    Args:
        constraints (list[tuple[str, str]]): A list of constraints tuple.
        provider (GeoSessionProvider): The provider for the given municipality.

    Returns:
        list[tuple[str, str]]: The list of location-aware constraints chunks and their source.
    """
    if len(constraints) == 0:
        return []

    SCALING_FACTOR = 0.05
    # retrieve documents
    constraints_chunks, constraints_sources = reduce(
        lambda res, c: ([*res[0], c[0]], [*res[1], c[1]]),
        constraints,
        ([], [])
    )

    prompt = [HumanMessage(content=PromptTemplate.from_template("""
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
        """).format(
        scaling_factor=SCALING_FACTOR,
        constraints="\n".join(f"<doc>{chunk}</doc>" for chunk in constraints_chunks),
    ))]
    # prompt the llm for the scaled
    # constraints specific for the
    # location
    response = await llm.ainvoke(prompt)
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
