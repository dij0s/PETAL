import re

from typing import Optional

from functools import reduce

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import  SystemMessage
from langgraph.config import get_stream_writer

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

async def guidelines_retriever(state):
    """
    Function for retrieving and augmenting the conversation state with relevant guidelines data.

    Args:
        state: The current conversation state to which we add the retrieved geo-context.

    Returns:
        dict: The updated conversation state.
    """
    writer = get_stream_writer()

    geocontext: Optional[GeoContextOutput] = state.geocontext
    if geocontext is None:
        geocontext = GeoContextOutput()

    router_state: RouterOutput = state.router
    if router_state.location is None or router_state.aggregated_query is None:
        raise ValueError("State is undefined")
    # start the instantiation of
    # the GeoSession for resident
    # count to reduce latency
    provider = GeoSessionProvider.get_or_create(router_state.location, 100, 1.0, with_residents_count=True)

    # retrieve relevant tools
    # and process constraints
    # for location-aware data
    writer({"type": "info", "content": "Retrieving effective guidelines..."})
    toolbox: ToolProvider = await ToolProvider.acreate(router_state.location)
    constraints = await toolbox.asearch_constraints(query=router_state.aggregated_query)
    writer({"type": "log", "content": "Found guidelines"})

    # process constraints
    writer({"type": "info", "content": "Processing guidelines..."})
    try:
        processed_constraints = await _process_constraints(constraints, provider)
    except RuntimeError:
        # location is not a proper
        # municipalty, enquire more
        # clarification by unsetting
        # the non-valid location
        router_state.location = None
        router_state.needs_clarification = True
        return {
            "router": router_state
        }
    # update context with
    # retrieved constraints
    # overwrite only as query
    # dependent
    geocontext.context_constraints = processed_constraints
    return {
        "geocontext": geocontext,
    }

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
    if (len(constraints) == 0):
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
