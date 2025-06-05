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

## Critical Guidelines

**IMPORTANT - Official Context**: The legislation and other relevant documents for effective energy planning define the actions and strategies that must be implemented to meet the requirements for the coming years and decades. Be sure to include this information, as these are the official guidelines from the state and country. **ALWAYS cite the source** when referencing legislative documents or official guidelines.

**User Preferences**: {memories_description}
Note: While preferences might reference a specific location, they should be understood as general user preferences.

---

## Your Task

**Response Requirements**:
- Answer using **only** the provided data
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
- Use appropriate headers (##, ###) to structure your response
- Use bullet points or numbered lists for key findings
- Include tables when comparing multiple data points
- Use **bold** for important values and findings
- Use *italics* for emphasis on policy recommendations
- **When citing legislative documents or official guidelines only, always include the source** in your response using the format: *(Source: [Document Name])*. There is no need to include the source for data points.

**Conclusion**:
End with a "## Recommended Next Steps" section suggesting one or more related analyses from the available data sources in the same category/categories "{categories}", phrased in a friendly and helpful way:
{related_tools_description}

---

Please respond in {lang}, ensuring all content is appropriately translated and formatted in markdown.
""")

# user_prompt_with_constraints = PromptTemplate.from_template("""
# You have already gathered the relevant data for the location "{location}". This data is provided below, with each entry including a description (explaining what the data represents and its units) and the corresponding value.

# Available data:
# {tools_data}

# User request: "{aggregated_query}"

# The legislation and other relevant documents for effective energy planning provided us with additional information on the matter. Make sure to include this information, as these are the guidelines from the state and country. Explain the goals and constraints in detail, without omitting any possible dates mentioned:
# {constraints}
# """)

# user_prompt_no_constraints = PromptTemplate.from_template("""
# You have already gathered the relevant data for the location "{location}". This data is provided below, with each entry including a description (explaining what the data represents and its units) and the corresponding value.

# Available data:
# {tools_data}

# User request: "{aggregated_query}"
# """)
#
user_prompt = PromptTemplate.from_template("""
## Data Summary for {location}

The following data has been gathered and is available for your analysis:

### Retrieved Data Points
{tools_data}
Note: A data point whose value is "0" indicates that no such data is produced, consumed, or present for the specified location; it does not mean that this data is unavailable elsewhere.

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

    last_human_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))
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
        "user_query": last_human_message,
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

    prompt = [
        SystemMessage(content=system_prompt.format(**prompt_args)),
        HumanMessage(content=user_prompt.format(**prompt_args))
    ]
    response = await llm.ainvoke(prompt)

    # update state with response
    # and push the new layers and
    # municipality's SFSO number
    # if there are any layers
    if len(layers) > 0:
        await provider.wait_until_sfso_ready()
        writer({"type": "layers", "layers": layers})
        writer({"type": "sfso_number", "sfso_number": provider.municipality_sfso_number})

    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)],
    }
