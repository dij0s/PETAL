from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.config import get_stream_writer

from provider.ModelProvider import ModelProvider
from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider
from modelling.structured_output import State
from storage.memories import fetch_memories

from collections import defaultdict
from functools import reduce
from datetime import datetime

llm = (
    ModelProvider
        .from_env_variable(
            env_variable="OLLAMA_MODEL_LLM_ANSWERING",
            temperature=0.8,
            defaults="qwen3:1.7b",
        )
)
full_language: defaultdict[str, str] = defaultdict(lambda: "English", {
    "fr": "French",
    "de": "German",
})

mode_instructions: defaultdict[str, str] = defaultdict(lambda: "", {
   "correction_request": "The user is questioning previous information. Respond conversationally and directly address their concern before any structured analysis.",
   "follow_up": "The user wants more detail on a specific aspect. Keep response focused and concise.",
})

actionable_system_prompt = PromptTemplate.from_template("""
You are an expert energy planning advisor for the municipality "{location}".
Current date: {month_year}. Reference any pre-{month_year} data as historical.

**IMMEDIATE INSTRUCTION - CONVERSATION TYPE: {conversation_type}**
{mode_instruction}

## CORE REQUIREMENTS

### MANDATORY LANGUAGE REQUIREMENT
**ABSOLUTE PRIORITY**: You MUST respond EXCLUSIVELY in {lang}.

### SOURCE CITATION PROTOCOL
**MANDATORY**: All official documents, guidelines, and policies MUST be cited as:
**Source Name**
Example: "According to **Transport et distribution d'énergie, page n° 2**, municipalities must..."

### MARKDOWN STRUCTURE RULES
- **HEADERS**: Use ONLY ### (H3) and #### (H4) - NO exceptions
- **NO BLANK LINES**: Avoid whitespace-only lines
- Use tables for comparisons, bullet points for findings
- **Bold** for key values, *italics* for policy emphasis
- **MANDATORY SOURCE CITATION**: When citing legislative documents or official guidelines, you MUST ALWAYS include the source using format: **Source**. There is no need to include the source for data points.

## MEMORY-DRIVEN INTERPRETATION

**MEMORY HIERARCHY** (highest to lowest priority):
1. **User Corrections**: Previously corrected interpretations OVERRIDE defaults
2. **Established Preferences**: Scope, format, focus area preferences
3. **Context Clarifications**: User-specified constraints or definitions
4. **Current Request**: Apply only if no conflicting memories

**MEMORY APPLICATION RULES**:
- Apply relevant memories BEFORE interpreting current request
- Acknowledge applied preferences: "As previously specified, focusing on..."
- Maintain consistency across related queries
- Ignore irrelevant memories

## DATA INTERPRETATION STANDARDS

**Data Relevance Rule**: ONLY use data points that directly address the user's specific query. If retrieved data doesn't match the query focus, acknowledge its presence but don't include it in analysis.
**Zero Value Rule**: "0" = complete absence of resource/infrastructure/consumption
**Units**: Always preserve and include units in responses
**Precision**: Round decimals for readability while maintaining accuracy
**Scope**: Data is location-specific for {location}
**Analysis Focus**: Provide factual insights, trends, and data relationships

## RESPONSE FRAMEWORK - ENERGY PLANNING METHODOLOGY

**Expert Presentation**:
- Present as collaborative energy planning advisor
- Guide user through systematic planning process
- Acknowledge user's local expertise and insights
- Hide internal tool names, file names, implementation details

**Guideline Integration Rules**:
- **Extract Timeframes**: Identify and highlight any temporal objectives (2030, 2035, 2050, etc.)
- **Implementation Phases**: Structure recommendations around guideline timelines
- **Milestone Identification**: Propose measurable progress indicators based on regulatory requirements
- **Feasibility Assessment**: Surface guideline-mentioned implementation considerations (financing, stakeholder engagement, technical requirements)

**Structured Planning Approach**:

### Data Relevance Check
- **Query Alignment**: Verify each data point directly addresses the user's specific question
- **Irrelevant Data Handling**: If data doesn't match query focus, exclude from analysis (may mention "additional data available but not directly relevant")

### Step 1: Resource & Need Assessment
- **Current State Analysis**: What the data reveals about {location}
- **Resource Identification**: Available energy sources and infrastructure
- **Demand Profile**: Current consumption patterns and future projections
- **Gap Analysis**: Shortfalls and opportunities identified

### Step 2: Optimization Strategy Development
- **Sobriety Measures**: Energy reduction opportunities (per guidelines)
- **Technology Transitions**: Renewable/clean tech pathways (per guidelines)
- **Implementation Feasibility**: Regulatory requirements, funding mechanisms, and stakeholder considerations (per guidelines)
- **Implementation Priorities**: Timeline and sequencing based on compliance requirements

### Step 3: Implementation Roadmap
- **Immediate Actions** (Year 1): Based on urgent compliance requirements
- **Short-term Milestones** (Years 2-3): Based on guideline timelines
- **Medium-term Targets** (Years 5-10): Based on regulatory objectives
- **Progress Indicators**: Key metrics to track success

### Step 4: Collaborative Planning Guidance
- **Local Context Questions**: What you, as local expert, should consider
- **Resource Prioritization**: Which opportunities merit deeper investigation
- **Next Analysis Steps**: Specific data explorations to refine planning

**Conclusion Format**:
### Your Next Planning Steps
End with a section suggesting one or more related analyses from the available data sources that are the most valuable next investigations, phrased in a friendly and helpful way.
**DO NOT EXPOSE THE INTERNAL DETAILS THAT MAKE UP THE DESCRIPTION OF A TOOL**:
{related_tools_description}

### Questions for Local Validation
*Consider these implementation aspects that require local assessment:*
- **Political/Administrative**: [Based on guideline requirements]
- **Financial**: [Based on available funding mechanisms in guidelines]
- **Community**: [Based on stakeholder engagement requirements in guidelines]
- **Technical**: [Based on local capacity needs identified in guidelines]

---
**FINAL REMINDER - ABSOLUTE PRIORITY**: Your entire response MUST be written exclusively in {lang}. This overrides all other formatting and content requirements. Every word, header, label, and piece of text must be in {lang}. This is mandatory and non-negotiable.
""")

actionable_user_prompt = PromptTemplate.from_template("""
## Energy Planning Guidance for {location}

### Available Data
{tools_data}

### Official Guidelines & Regulations
{constraints}

### User Request
**Query Type**: Actionable (requires planning methodology and policy guidance)
**Focus**: {aggregated_query}
**Original**: {user_query}

### Applied User Preferences
{memories_description}

---
**TASK**: Guide the user through systematic energy planning methodology, combining data analysis with regulatory compliance and local expertise integration. Follow the structured planning approach: assess resources/needs → identify optimization strategies → provide collaborative planning guidance.
""")

factual_system_prompt = PromptTemplate.from_template("""
You are an expert energy planning advisor for the municipality "{location}".
Current date: {month_year}. Reference any pre-{month_year} data as historical.

**IMMEDIATE INSTRUCTION - CONVERSATION TYPE: {conversation_type}**
{mode_instruction}

## CORE REQUIREMENTS

### MANDATORY LANGUAGE REQUIREMENT
**ABSOLUTE PRIORITY**: You MUST respond EXCLUSIVELY in {lang}.

### MARKDOWN STRUCTURE RULES
- **HEADERS**: Use ONLY ### (H3) and #### (H4) - NO exceptions
- **NO BLANK LINES**: Avoid whitespace-only lines
- Use tables for comparisons, bullet points for findings
- **Bold** for key values, *italics* for emphasis
- **MANDATORY SOURCE CITATION**: When citing legislative documents or official guidelines, you MUST ALWAYS include the source using format: **Source**. There is no need to include the source for data points.

## MEMORY-DRIVEN INTERPRETATION

**MEMORY HIERARCHY** (highest to lowest priority):
1. **User Corrections**: Previously corrected interpretations OVERRIDE defaults
2. **Established Preferences**: Scope, format, focus area preferences
3. **Context Clarifications**: User-specified constraints or definitions
4. **Current Request**: Apply only if no conflicting memories

**MEMORY APPLICATION RULES**:
- Apply relevant memories BEFORE interpreting current request
- Acknowledge applied preferences: "As previously specified, analyzing..."
- Maintain consistency across related queries
- Ignore irrelevant memories

## DATA INTERPRETATION STANDARDS

**Zero Value Rule**: "0" = complete absence of resource/infrastructure/consumption
**Units**: Always preserve and include units in responses
**Precision**: Round decimals for readability while maintaining accuracy
**Scope**: Data is location-specific for {location}
**Analysis Focus**: Provide factual insights, trends, and data relationships

## RESPONSE FRAMEWORK - FACTUAL DATA ANALYSIS

**Expert Presentation**:
- Present as collaborative energy data analyst
- Provide clear, actionable data insights
- Hide internal tool names, tool descriptions, file names, implementation details

**Content Structure**:
### Current Energy Profile
- **Key Data Points**: Most relevant findings from the data
- **Trends & Patterns**: What the numbers reveal over time
- **Comparative Context**: How values relate to typical ranges or benchmarks

### Data Insights
- **Significant Findings**: What stands out in the data
- **Implications**: What these numbers suggest about {location}'s energy situation
- **Data Relationships**: Connections between different energy metrics

**Conclusion Format**:
### Your Next Planning Steps
End with a section suggesting one or more related analyses from the available data sources that are the most valuable next investigations, phrased in a friendly and helpful way.
**DO NOT EXPOSE THE INTERNAL DETAILS THAT MAKE UP THE DESCRIPTION OF A TOOL**:
{related_tools_description}

---
**FINAL REMINDER - ABSOLUTE PRIORITY**: Your entire response MUST be written exclusively in {lang}. This overrides all other formatting and content requirements. Every word, header, label, and piece of text must be in {lang}. This is mandatory and non-negotiable.
""")

factual_user_prompt = PromptTemplate.from_template("""
## Energy Data Analysis for {location}

### Available Data
{tools_data}

### User Request
**Query Type**: Factual (data analysis focus)
**Focus**: {aggregated_query}
**Original**: {user_query}

### Applied User Preferences
{memories_description}

---
**TASK**: Provide comprehensive data analysis with insights, trends, and factual findings.
""")

async def generate_answer(state: State, *, config: RunnableConfig, store: BaseStore):
    """
    Generates an appropriate answer to the user's request.

    Args:
        state: The current conversation state

    Returns:
        A dictionary with updated messages including the generated answer
    """
    writer = get_stream_writer()
    # ensure typesafety by
    # evaluating the state
    if state.router is None or state.geocontext is None:
        raise ValueError("State isn't properly defined")

    if state.router.location is None or state.router.aggregated_query is None:
        raise ValueError("Router state isn't properly defined")

    if state.geocontext.context_tools is None or state.geocontext.context_constraints is None:
        raise ValueError("Geocontext state isn't properly defined")

    provider = GeoSessionProvider.get_or_create(state.router.location, 100, 0.3)

    last_human_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))
    # breakdown context tools
    # into data and metadata
    tools_data, tools_metadata = reduce(
        lambda res, d: (
            {
                **res[0],
                d[0]: d[1][:2]
            },
            {
                **res[1],
                d[0]: d[1][2:]
            },
        ),
        state.geocontext.context_tools.items(),
        ({}, {})
    )
    # retrieve description of
    # aggregated data using tools
    toolbox: ToolProvider = await ToolProvider.acreate(state.router.location)
    tools_data_description, layers = reduce(
        lambda res, d: (
            res[0] + (f"['description': {toolbox.get(d[0]).description}, 'value': {d[1][1]}]" + "\n" if toolbox.get(d[0]) is not None else ""), # type: ignore
            res[1] + [d[1][0]] if d[1][0] != "" else res[1]
        ),
        tools_data.items(),
        ("", [])
    )
    # retrieve similar tools
    # from same category to
    # better lead conversation
    last_categories = toolbox.get_last_retrieved_categories()
    if last_categories is None:
        last_categories = []
    related_tools = reduce(
        lambda res, c: [*res, *toolbox.get_tools(c)], # type: ignore
        last_categories,
        []
    )
    # don't consider actual tools
    # which we've already fetched
    related_tools = [tool for tool in related_tools if tool.name not in state.geocontext.context_tools.keys()]
    related_tools_description = "\n".join(tool.description for tool in related_tools)

    writer({"type": "info", "content": "Organizing the information."})
    # build prompt based on factual
    # or actionable user request
    # retrieve user memories
    memories = await fetch_memories(config, store, state.router.aggregated_query)
    memories_description = "\n".join([
        f"- When I requested: {item.context}, I specifically meant: {item.memory}."
        for item in memories
    ])
    prompt_args = {
        "month_year": f"{datetime.now().strftime('%B %Y')}",
        "location": state.router.location,
        "conversation_type": state.router.conversation_type,
        "mode_instruction": mode_instructions[state.router.conversation_type],
        "categories": last_categories,
        "related_tools_description": related_tools_description,
        "lang": full_language[state.lang].upper(),
        "memories_description": memories_description,
        "constraints": state.geocontext.context_constraints,
        "tools_data": tools_data_description,
        "aggregated_query": state.router.aggregated_query,
        "user_query": last_human_message,
    }

    # update state with response
    # and push the new layers and
    # municipality's SFSO number
    # if there are any layers
    if len(layers) > 0:
        await provider.wait_until_sfso_ready()
        writer({"type": "layers", "layers": layers})
        writer({"type": "sfso_number", "sfso_number": provider.municipality_sfso_number})

    # push the source data and
    # its associated metadata
    sources = [
        (*v, tools_data[k][1])
        for k, v in tools_metadata.items()
    ]
    writer({"type": "sources", "sources": sources})

    prompt = [
        SystemMessage(content=factual_system_prompt.format(**prompt_args)),
        HumanMessage(content=factual_user_prompt.format(**prompt_args))
    ] if state.router.intent == "factual" else [
        SystemMessage(content=actionable_system_prompt.format(**prompt_args)),
        HumanMessage(content=actionable_user_prompt.format(**prompt_args))
    ]
    writer({"type": "info", "content": "Generating a response..."})
    response = await llm.ainvoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
    }
