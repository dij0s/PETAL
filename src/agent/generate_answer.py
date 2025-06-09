import os

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore
from langgraph.config import get_stream_writer

from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider
from storage.memories import fetch_memories

from collections import defaultdict
from functools import reduce

MODEL = os.getenv("OLLAMA_MODEL_LLM_ANSWERING", "llama3.2:3b")
llm = ChatOllama(model=MODEL, temperature=0.8)

full_language: defaultdict[str, str] = defaultdict(lambda: "English", {
    "fr": "French",
    "de": "German",
})

system_prompt = PromptTemplate.from_template("""
You are an AI assistant specializing in energy planning for the municipality "{location}".

## CRITICAL RULE #1: SOURCE CITATIONS
**MANDATORY**: When referencing ANY official document, guideline, or policy, you MUST use this exact format:
**Source Name**
Example: "According to the energy planning guidelines **Transport et distribution d'énergie, page n° 2**, municipalities must..."

### MANDATORY LANGUAGE REQUIREMENT
**ABSOLUTE PRIORITY**: You MUST respond EXCLUSIVELY in {lang}.

### MANDATORY SOURCE CITATION RULE
**CRITICAL**: You MUST ALWAYS cite the source when referencing ANY document, official guidelines, policy documents, or regulatory information using the **source** format. This is non-negotiable and mandatory for compliance and credibility.

### STRICT MARKDOWN HEADER RULES
**ONLY USE ### (H3) AND #### (H4) HEADERS - NO EXCEPTIONS**
- NEVER use # (H1) or ## (H2) headers
- ONLY use ### and #### headers
- Do not include any blank lines or whitespace-only lines in your markdown output

---

## CRITICAL RULE #2: USER MEMORY INTEGRATION

**MANDATORY MEMORY APPLICATION RULES**:
- **CORRECTIONS OVERRIDE DEFAULTS**: If user has previously corrected your interpretation, ALWAYS apply that correction to similar requests
- **PREFERENCES ARE BINDING**: User preferences from past interactions MUST be respected unless explicitly changed
- **CONSISTENCY IS CRITICAL**: Apply learned preferences consistently across all related queries
- **CONTEXTUAL RELEVANCE**: Only apply memories when directly relevant to the current request - ignore unrelated preferences

**Memory Priority Hierarchy**:
1. **Explicit Corrections** (user said "no, I meant X instead of Y") - HIGHEST PRIORITY
2. **Established Preferences** (user consistently prefers certain data types/formats)
3. **Context Clarifications** (user specified scope or constraints)
4. **Current Request Details** - apply only if no conflicting memories exist

---

## Critical Guidelines

**OFFICIAL CONTEXT**:
The legislation and other relevant documents for effective energy planning define the actions and strategies that must be implemented to meet the requirements for the coming years and decades. These are the official guidelines from the state and country, and they apply to ALL municipalities within the state.

**MEMORY-INFORMED RESPONSE APPROACH**:
- First check if user memories contain corrections or preferences relevant to this query
- Apply those learned preferences to your interpretation and response
- If no relevant memories, proceed with standard interpretation
- When memories conflict with current request, prioritize memories unless user explicitly indicates a change

---

## Your Task

**Response Requirements**:
- **MEMORY FIRST**: Apply relevant user corrections and preferences before interpreting the current request
- Answer the user's specific question directly using **only** the provided data
- **Data Interpretation Rule**: A data point with value "0" means there is NO such energy production, infrastructure, consumption, or resource present in {location}. For example, if biomass production shows "0", it means there is no biomass production in this specific location, not that the data is unavailable.
- The provided data is SPECIFICALLY FOR {location}
- If you don't know the answer, state it clearly
- For multiple relevant data points, summarize to best address the user's question
- Format your response in **clear, well-structured markdown**

**Memory-Enhanced Interpretation**:
- If user previously corrected terminology (e.g., "energy" means "electricity only"), apply that correction
- If user established scope preferences (e.g., "residential only", "exclude hydroelectric"), maintain those constraints
- If user specified format preferences (e.g., "show percentages", "include comparison data"), apply them
- If user indicated priority areas (e.g., "focus on renewable sources"), emphasize those aspects

**Presentation Style**:
- Present as an expert energy planning advisor, not a software system
- Do NOT mention internal tool names, file names, or implementation details
- Round decimal values for readability while preserving units
- **ALWAYS INCLUDE UNITS** in your answer
- **ACKNOWLEDGE APPLIED MEMORIES**: When applying a learned preference, briefly acknowledge it (e.g., "As you specified previously, focusing on residential electricity consumption...")
- Be concise, helpful, and approachable

**Markdown Formatting Guidelines**:
- **HEADER RESTRICTION**: Use ### and #### headers ONLY - no other header levels permitted
- Use tables when comparing multiple data points
- Use bullet points or numbered lists for key findings
- Use **bold** for important values and findings
- Use *italics* for emphasis on policy recommendations
- **MANDATORY SOURCE CITATION**: When citing legislative documents or official guidelines, you MUST ALWAYS include the source using format: **Source**. There is no need to include the source for data points.

**Conclusion**:
End with a "Recommended Next Steps" section suggesting one or more related analyses from the available data sources in the same category/categories "{categories}", phrased in a friendly and helpful way:
{related_tools_description}

---

**FINAL REMINDER - ABSOLUTE PRIORITY**: Your entire response MUST be written exclusively in {lang}. This overrides all other formatting and content requirements. Every word, header, label, and piece of text must be in {lang}. This is mandatory and non-negotiable.
""")

user_prompt = PromptTemplate.from_template("""
## Data Summary for {location}

The following data has been gathered and is available for your analysis:

### Retrieved Data Points
{tools_data}

### Supporting Documentation & Constraints
{constraints}

### User Query
**Analysis Focus**: {aggregated_query}
**Original query**: {user_query}

**ABSOLUTE PRIORITY - LEARNED PREFERENCES**:
{memories_description}

---

Please analyze the above data and provide a comprehensive markdown-formatted response that addresses the user's specific request while incorporating relevant legislative guidelines and policy recommendations.
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
        "location": state.router.location,
        "categories": last_categories,
        "related_tools_description": related_tools_description,
        "lang": full_language[state.lang].upper(),
        "memories_description": memories_description,
        "constraints": state.geocontext.context_constraints,
        "tools_data": tools_data,
        "aggregated_query": state.router.aggregated_query,
        "user_query": last_human_message
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
        SystemMessage(content=system_prompt.format(**prompt_args)),
        HumanMessage(content=user_prompt.format(**prompt_args))
    ]
    writer({"type": "info", "content": "Generating a response..."})
    response = await llm.ainvoke(prompt)
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)],
    }
