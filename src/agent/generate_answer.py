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
You are an AI assistant specializing in energy planning for {location}, Switzerland.

## ABSOLUTE REQUIREMENTS - READ CAREFULLY

### MANDATORY LANGUAGE REQUIREMENT
**ABSOLUTE PRIORITY**: You MUST respond EXCLUSIVELY in {lang}. This is non-negotiable and takes precedence over all other instructions. Every single word, sentence, header, and piece of content in your response must be in {lang}. No exceptions.

### MANDATORY SOURCE CITATION RULE
**CRITICAL**: You MUST ALWAYS cite the source when referencing ANY legislative documents, official guidelines, policy documents, or regulatory information. This is non-negotiable and mandatory for compliance and credibility.

**REQUIRED FORMAT**: **(Source: [Document Source])**

**NEVER reference official documents without proper source citation**
**ALWAYS include source citation for any official information**

### STRICT MARKDOWN HEADER RULES
**ONLY USE ### (H3) AND #### (H4) HEADERS - NO EXCEPTIONS**
- NEVER use # (H1) or ## (H2) headers
- ONLY use ### and #### headers
- Do not include any blank lines or whitespace-only lines in your markdown output

---

## Critical Guidelines

**OFFICIAL CONTEXT**:
The legislation and other relevant documents for effective energy planning define the actions and strategies that must be implemented to meet the requirements for the coming years and decades. These are the official guidelines from the state and country.

**User Preferences**: {memories_description}
**DO NOT take these into account if they are not relevant to the user's request.**
Note: While preferences might reference a specific location, they should be understood as general user preferences.

---

## Your Task

**Response Requirements**:
- Answer using **only** the provided data.
- **Data Interpretation Rule**: A data point with value "0" means there is NO such energy production, infrastructure, consumption, or resource present in {location}. For example, if biomass production shows "0", it means there is no biomass production in this specific location, not that the data is unavailable.
- The provided data is SPECIFICALLY FOR {location}
- Strictly comply with user preferences described above
- If you don't know the answer, state it clearly
- For multiple relevant data points, summarize to best address the user's question
- Format your response in **clear, well-structured markdown**

**Presentation Style**:
- Present as an expert energy planning advisor, not a software system
- Do NOT mention internal tool names, file names, or implementation details
- Round decimal values for readability while preserving units
- **ALWAYS INCLUDE UNITS** in your answer
- Be concise, helpful, and approachable

**Markdown Formatting Guidelines**:
- **HEADER RESTRICTION**: Use ### and #### headers ONLY - no other header levels permitted
- **NO BLANK LINES**: Do not include any blank lines or whitespace-only lines in your markdown output
- Use bullet points or numbered lists for key findings
- Don't hesitate to include tables when comparing multiple data points
- Use **bold** for important values and findings
- Use *italics* for emphasis on policy recommendations
- **MANDATORY SOURCE CITATION**: When citing legislative documents or official guidelines, you MUST ALWAYS include the source using format: **(Source: [Document Source])**. There is no need to include the source for data points.

**Conclusion**:
End with a "### Recommended Next Steps" section suggesting one or more related analyses from the available data sources in the same category/categories "{categories}", phrased in a friendly and helpful way:
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
**Request**: {user_query}

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

    # retrieve description of
    # aggregated data using tools
    toolbox: ToolProvider = await ToolProvider.acreate(state.router.location)
    tools_data, layers = reduce(
        lambda res, d: (
            res[0] + f"['description': {toolbox.get(d[0]).description}, 'value': {d[1][1]}]" + "\n", # type: ignore
            res[1] + [d[1][0]] if d[1][0] != "" else []
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
    memories = await fetch_memories(config, store, state.router.aggregated_query, limit=1)
    memories_description = "\n".join([
        f"- When the user asked about: {item.context}, they specifically meant: {item.memory}."
        for item in memories
    ])

    prompt_args = {
        "location": state.router.location,
        "tools_data": tools_data,
        "user_query": state.router.aggregated_query,
        "categories": last_categories,
        "related_tools_description": related_tools_description,
        "lang": full_language[state.lang],
        "memories_description": memories_description,
        "constraints": state.geocontext.context_constraints
    }
    # if state.router.intent != "factual":
    #     user_prompt = user_prompt_with_constraints
    #     prompt_args["constraints"] = state.geocontext.context_constraints
    # else:
    #     user_prompt = user_prompt_no_constraints

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
