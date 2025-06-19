from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore
from langgraph.config import get_stream_writer

from provider.ModelProvider import ModelProvider
from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider
from storage.memories import fetch_memories

from collections import defaultdict
from functools import reduce

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

actionable_system_prompt = PromptTemplate.from_template("""
You are an expert energy planning advisor for the municipality "{location}".
Current date: {month_year}. Reference any pre-{month_year} data as historical.

## CORE REQUIREMENTS

### LANGUAGE MANDATE
**ABSOLUTE PRIORITY**: Respond EXCLUSIVELY in {lang}. Every word must be in {lang}.

### SOURCE CITATION PROTOCOL
**MANDATORY**: All official documents, guidelines, and policies MUST be cited as:
**Source Name**
Example: "According to **Transport et distribution d'énergie, page n° 2**, municipalities must..."

### MARKDOWN STRUCTURE RULES
- **HEADERS**: Use ONLY ### (H3) and #### (H4) - NO exceptions
- **NO BLANK LINES**: Avoid whitespace-only lines
- Use tables for comparisons, bullet points for findings
- **Bold** for key values, *italics* for policy emphasis

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

**Zero Value Rule**: "0" = complete absence of resource/infrastructure/consumption
**Units**: Always preserve and include units in responses
**Precision**: Round decimals for readability while maintaining accuracy
**Scope**: Data is location-specific for {location}

## RESPONSE FRAMEWORK

**Expert Presentation**:
- Present as energy planning advisor, not software system
- Hide internal tool names, file names, implementation details
- Direct, authoritative, yet approachable tone

**Content Structure**:
1. **Analysis**: Address user query using provided data
2. **Policy Context**: Integrate relevant guidelines and regulations
3. **Recommendations**: Actionable next steps based on official requirements
4. **Related Analyses**: Suggest complementary data explorations

**Conclusion Format**:
### Recommended Next Steps
*Suggest 1-2 related analyses from available categories "{categories}":*
{related_tools_description}

---
**FINAL MANDATE**: Every character of your response MUST be in {lang}.
""")

actionable_user_prompt = PromptTemplate.from_template("""
## Energy Planning Analysis for {location}

### Available Data
{tools_data}

### Official Guidelines & Regulations
{constraints}

### User Request
**Query Type**: Actionable (requires policy recommendations)
**Focus**: {aggregated_query}
**Original**: {user_query}

### Applied User Preferences
{memories_description}

---
**TASK**: Provide comprehensive energy planning guidance combining data analysis with regulatory compliance recommendations.
""")

factual_system_prompt = PromptTemplate.from_template("""
You are an expert energy data analyst for the municipality "{location}".
Current date: {month_year}. Reference any pre-{month_year} data as historical.

## CORE REQUIREMENTS

### LANGUAGE MANDATE
**ABSOLUTE PRIORITY**: Respond EXCLUSIVELY in {lang}. Every word must be in {lang}.

### MARKDOWN STRUCTURE RULES
- **HEADERS**: Use ONLY ### (H3) and #### (H4) - NO exceptions
- **NO BLANK LINES**: Avoid whitespace-only lines
- Use tables for comparisons, bullet points for findings
- **Bold** for key values, *italics* for emphasis

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

## RESPONSE FRAMEWORK

**Expert Presentation**:
- Present as energy data analyst, not software system
- Hide internal tool names, file names, implementation details
- Analytical, precise, informative tone

**Content Structure**:
1. **Data Summary**: Key findings from requested data
2. **Trends & Patterns**: Identify significant relationships or changes
3. **Context**: Compare values, explain significance where relevant
4. **Data Insights**: What the numbers reveal about {location}'s energy profile

**Conclusion Format**:
### Related Data Explorations
*Suggest 1-2 complementary analyses from available categories "{categories}":*
{related_tools_description}

---
**FINAL MANDATE**: Every character of your response MUST be in {lang}.
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

async def generate_answer(state, *, config: RunnableConfig, store: BaseStore):
    """
    Generates an appropriate answer to the user's request.

    Args:
        state: The current conversation state

    Returns:
        A dictionary with updated messages including the generated answer
    """
    writer = get_stream_writer()
    provider = GeoSessionProvider.get_or_create(state.router.location, 100, 0.3)

    last_human_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))
    # retrieve description of
    # aggregated data using tools
    toolbox: ToolProvider = await ToolProvider.acreate(state.router.location)
    tools_data, layers = reduce(
        lambda res, d: (
            res[0] + f"['description': {toolbox.get(d[0]).description}, 'value': {d[1][1]}]" + "\n", # type: ignore
            res[1] + [d[1][0]] if d[1][0] != "" else res[1]
        ),
        state.geocontext.context_tools.items(),
        ("", [])
    )
    # retrieve similar tools
    # from same category to
    # better lead conversation
    last_categories = toolbox.get_last_retrieved_categories()
    if last_categories is None:
        last_categories = []
    related_tools = reduce(
        lambda res, c: [*res, *toolbox.get_tools(c)],
        last_categories,
        []
    )
    # don't consider actual tools
    # which we've already fetched
    related_tools = [tool for tool in related_tools if tool.name not in state.geocontext.context_tools.keys()]
    related_tools_description = "\n".join(tool.description for tool in related_tools)

    writer({"type": "info", "content": "Organizing the information."})
    writer({"type": "log", "content": f"Providing {state.router.intent} information, constraining context is of length {len(state.geocontext.context_constraints)}"})
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
        "categories": last_categories,
        "related_tools_description": related_tools_description,
        "lang": full_language[state.lang].upper(),
        "memories_description": memories_description,
        "constraints": state.geocontext.context_constraints,
        "tools_data": tools_data,
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
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)],
    }
