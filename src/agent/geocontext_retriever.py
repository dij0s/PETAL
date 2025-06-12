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
    constraints_chunks, constraints_sources = reduce(
        lambda res, c: ([*res[0], c[0]], [*res[1], c[1]]),
        awaited_constraints,
        ([], [])
    )

    # prompt = [HumanMessage(content=PromptTemplate.from_template("""
    #     You are a text processor. Your job is to scale energy numbers in the following documents.

    #     Instructions:
    #     - For each document, find numbers immediately followed by GWh, MWh, MW, kWh, kW, or GW (case-insensitive).
    #     - Multiply ONLY those numbers by {scaling_factor} and replace them in the text, rounded to 1 decimal place.
    #     - DO NOT scale percentages, dates, or any other numbers.
    #     - DO NOT add any explanations, notes, or comments.
    #     - DO NOT change any other part of the text.
    #     - Return the processed documents, separated by <doc>.

    #     Input documents:
    #     {constraints}

    #     Output:
    #     Return the same documents, in the same order, separated by <doc>. Only the relevant energy numbers should be changed.
    #     """).format(
    #     scaling_factor=SCALING_FACTOR,
    #     constraints=".\n<doc>\n".join(constraints_chunks),
    # ))]
    # # prompt the llm for the scaled
    # # constraints specific for the
    # # location
    # response = await llm.ainvoke(prompt)
    # # extract the documents
    # # from the response and
    # # return original ones
    # # on fallback
    # try:
    #     document_pattern = re.compile(r"<doc>(.*?)</doc>", re.DOTALL)
    #     processed_constraints = [doc.strip() for doc in document_pattern.findall(response.content)] # type: ignore
    #     temp = reduce(
    #         lambda res, cs: [*res, (cs[0], cs[1])],
    #         zip(processed_constraints, constraints_sources),
    #         []
    #     )
    #     print(temp)
    #     print([cs for cs in zip(constraints_chunks, constraints_sources)])
    #     return temp
    # except:
    #     return [cs for cs in zip(constraints_chunks, constraints_sources)]
    #
    temp = "\b(\d+(?:[',.]\d*)*(?:\.\d+)?)\s?[A-Za-z]?Wh?\b"
    print(awaited_constraints)
    docs = [[('### Key Insights\n- **Energy Mix Transition**: The energy mix is shifting away from fossil fuels towards renewable sources. In 2010, fossil fuels covered 65% of energy needs, but by 2020, this is projected to decrease to 59%.\n- **Hydroelectricity Dominance**: Hydroelectric power remains the primary source of energy, contributing significantly to the overall production, with a target increase in production of 1,400 GWh by 2020.\n- **Renewable Energy Growth**: There is a focus on increasing the use of renewable energy sources such as solar, wind, wood, biomass, and heat recovery. The goal is to increase the production of indigenous renewable energy by 1,400 GWh by 2020.\n- **Energy Efficiency**: Efforts are being made to reduce overall energy consumption by 5% compared to 2010, despite expected population and economic growth.\n- **Local Energy Production**: There is a strategic emphasis on keeping local energy production in the hands of local authorities and enterprises, particularly for electricity generation.', 'Approvisionnement en énergie, page n° 2'), ('### Key Insights\n- The document emphasizes the importance of transitioning to a 100% renewable energy supply by 2060 in the Valais region.\n- The Valais is highlighted as a significant producer of renewable energy resources such as water, solar, wind, wood, etc., which can contribute substantially to the national energy supply.\n- There is a strong focus on the need for a sustainable, efficient, and renewable energy policy that aligns with the country’s broader energy strategy.\n- The document underscores the ambition to achieve full energy independence through local renewable resources, with the potential for the Valais to cover its entire energy needs by 2060.\n- The transition to renewable energy is seen as a collective effort involving various stakeholders including communities, producers, distributors, and other industry actors.', "Vision 2060 et objectifs 2035, Valais, Terre d'énergies, page n° 3"), ('### Key Insights\n- The canton and communes are mandated to follow exemplary standards in all their activities related to energy efficiency.\n- Higher energy efficiency requirements are set for buildings owned by or partially funded by the canton or communes, with non-compliant buildings ineligible for subsidies.\n- The Council of State establishes stricter energy standards for infrastructure, the cantonal fleet, and electrical equipment.\n- A comprehensive energy excellence plan is developed covering all cantonal activities and recommending energy improvements to companies where the canton is a stakeholder.\n- The canton ensures exemplary energy management of its real estate portfolio, including collecting, publishing, and utilizing consumption data.\n- New public lighting must be designed, implemented, operated, and maintained to be energy-efficient and environmentally friendly, with power and duration reduced to necessary levels for safety and specific usage.\n- The goal for cantonal buildings and installations is to achieve heat supply without fossil fuels by 2035, use electricity efficiently, and utilize on-site renewable energy potentials.', "Loi sur l'énergie (LcEne), page n° 13"), ("### Key Insights\n- The primary focus is on reducing energy consumption for heat production by 23% between 2015 and 2035, aiming to reach 2,790 GWh/a.\n- Key strategies include:\n- Increasing the use of renewable energy sources and heat rejection.\n- Improving building insulation and efficiency.\n- Enhancing the management of secondary residences' energy needs.\n- Installing heat recovery systems.\n- Transition patterns indicate a shift towards renewable energy and improved energy efficiency in buildings.", "Vision 2060 et objectifs 2035, Valais, Terre d'énergies, page n° 30")],
        [('### Key Insights\n- **Capacity Expansion Needed**: The current transmission network in Valais is insufficient to handle new production capacities, necessitating increased capacity and adaptation.\n- **Replacement of Lignes 220 kV**: Several lines at 220 kV, nearing their capacity limits, will need to be replaced with 380 kV lines, such as the "Chamoson-Chippis" line, which is not included in the PSE.\n- **Network Restructuring**: The 125 kV network will be progressively phased out, downgraded, or partially replaced by higher-voltage lines (380, 220, or 65 kV).\n- **Heat Distribution Networks**: There is a significant development of heat distribution networks in Valais, aiming for a 210 GWh increase in heat distributed by 2020, contributing 8% to the canton’s target of 6,111 GWh under the 2050 energy strategy.', "Transport et distribution d'énergie, page n° 2"), ("### Key Insights\n- The document outlines various activities related to coordinating civil functions of the Sion airport, including financial support, spatial management, and noise pollution resolution.\n- Measures are put in place to maintain and develop civil functions, particularly focusing on financial aspects and spatial management.\n- There is a focus on maintaining a limited level of noise pollution while supporting military presence at the airport.\n- Coordination with the OFAC (Office Fédéral de l'Aviation Civile) is emphasized for managing air navigation obstacles and ensuring rational and efficient use of mountain airspace.\n- Specific measures include repositioning landing areas, defining approach sectors, restricting flights based on seasons and times, and exploring alternative solutions like creating restricted zones for free movement.\n- Communes participate in planning processes through the PSIA coordination protocol and ensure spatial planning of affected areas, including noise abatement zones and height restrictions near approach zones.\n- Synergies with existing infrastructure are highlighted, such as agricultural, industrial, and sports activities, as well as potential renaturations.", 'Infrastructures aéronautiques, page n° 4'), ('### Key Insights\n- **Maintained Installations**: The concept of stationnement 2016 maintains key military installations such as the Police Military Post in Sierre, the Armory in Sion, and the Simplon, Visp, and Sion logistics centers.\n- **Closure of Major Installations**: The closure of Brig Barracks, the Armory and infrastructure centers in St-Maurice-Lavey, and the Airfield in Sion is planned.\n- **Secondary Installations**: The concept retains 25 cantonments, 3 logistics centers, 6 firing/exercise sites, and 3 crossing points but plans the closure of 7 cantonments, 3 logistics/engagement centers, and 12 firing/exercise sites.\n- **Coordination Principles**: The document outlines five principles for coordination, emphasizing the need for information sharing, prioritizing civilian interests, favoring local hosting of military activities, allowing for repurposing of decommissioned military facilities, and ensuring environmental compliance during facility changes.', 'Installations militaires, page n° 2')],
        [('The provided image is a static visual with no charts, plots, tables, or quantitative data to analyze for insights, trends, or strategic implications. It features a scenic view of a mountainous landscape with a lake and a winding road, overlaid with a red gradient in the lower left corner. The image also includes contact information for the "Département des finances et de l\'énergie" and "Service de l\'énergie et des forces hydrauliques" located in Sion, Switzerland.', "Vision 2060 et objectifs 2035, Valais, Terre d'énergies, page n° 64")]]
    return awaited_constraints

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
