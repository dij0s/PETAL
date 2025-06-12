import os
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

MODEL = os.getenv("OLLAMA_MODEL_LLM_ROUTING", "llama3.2:3b")
llm = ChatOllama(model=MODEL, temperature=0).with_structured_output(ConstraintsOutput, method="function_calling")

parser = PydanticStreamOutputParser(pydantic_object=ConstraintsOutput, diff=True)

processing_prompt = PromptTemplate.from_template("""
You are an expert energy planning consultant transforming cantonal (state-level) energy guidelines and key figures into practical municipal-level guidance.

MUNICIPALITY CONTEXT:
- Municipality: {location}
- Population scaling factor: {scaling_factor}
- Task: Convert cantonal guidelines into actionable municipal energy planning guidance

TRANSFORMATION RULES:

1. **ENERGY TARGETS & QUANTITIES**:
   - Scale absolute energy values (GWh, MWh, MW) by multiplying by {scaling_factor}
   - Example: "210 GWh cantonal target" → "10.5 GWh municipal contribution"

2. **INFRASTRUCTURE & REGIONAL CONTEXT**:
   - Keep technical specifications (380 kV, 220 kV ratings) unchanged
   - Explain how regional infrastructure affects {location}

3. **PERCENTAGES, DATES & RATIOS**:
   - Keep all percentages unchanged (23% reduction, 8% contribution, etc.)
   - Preserve all target years and timeframes
   - Maintain efficiency ratios and performance standards

ORIGINAL CANTONAL GUIDELINES:
{constraints}

OUTPUT REQUIREMENTS:
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
            writer({"type": "info", "content": "Retrieving tools..."})
            toolbox: ToolProvider = await ToolProvider.acreate(router_state.location)
            tools, constraints = await toolbox.asearch(query=router_state.aggregated_query, max_n_tools=5, k_tools=10, process_constraints=partial(_process_constraints, provider=provider))
            writer({"type": "log", "content": "I FOUND THEM!"})
            # filter out tools whose
            # data we already have
            tools = [tool for tool in tools if tool.name not in geocontext.context_tools.keys()]
            # invoke chosen tools
            # and update context state
            if len(tools) > 0:
                writer({"type": "info", "content": "Fetching data from retrieved tools..."})
                tool_data = await _ainvoke_tools(tools)
                geocontext.context_tools = {**geocontext.context_tools, **tool_data}
            else:
                # needed data is already retrieved
                writer({"type": "info", "content": "We already have them!"})
            # update context with
            # retrieved constraints
            # overwrite only as query
            # dependent
            geocontext.context_constraints = constraints
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

async def _process_constraints(constraints: Awaitable[list[tuple[str, str]]], provider: GeoSessionProvider) -> list[tuple[str, str]]:
    """
    Processes a list of constraints asynchronously.

    This function processes the state-wide constraints and returns the location-aware constraints.

    Args:
        constraints (Awaitable[list[tuple[str, str]]]): An awaitable that yields a list of constraint tuples.
        provider (GeoSessionProvider): The provider for the given municipality.

    Returns:
        list[tuple[str, str]]: The list of location-aware constraints chunks and their source.
    """
    # TODO
    # retrieve from provider itself
    SCALING_FACTOR = 0.05
    # retrieve documents
    awaited_constraints = await constraints
    constraints_chunks = reduce(
        lambda res, c: [*res, c[0]],
        awaited_constraints,
        []
    )

    prompt = [SystemMessage(content=processing_prompt.format(
        location=provider.municipality_name, # type: ignore
        scaling_factor=SCALING_FACTOR,
        constraints="\n".join(constraints_chunks),
        format_description_llm=parser.get_description()
    ))]
    updated_constraints = reduce(
        lambda res, cs: [*res, (cs[0], cs[1])],
        zip(
            (await llm.ainvoke(prompt)).documents,
            map(lambda c: c[1], awaited_constraints)
        ), # type: ignore
        []
    )

    print(awaited_constraints)
    print("original")
    print(updated_constraints)
    return updated_constraints

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
