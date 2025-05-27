from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer

from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider

from functools import reduce

llm = ChatOllama(model="llama3.2:3b", temperature=0.8)

answer_prompt = PromptTemplate.from_template("""
You are an AI assistant specializing in energy planning for {location}.

You have already gathered the relevant data for the user's requested location "{location}". This data is provided below, with each entry including a description (explaining what the data represents and its units) and the corresponding value.

Available data:
{tool_data}

User request: "{aggregated_query}"

Additionally, here are some related analyses or data sources that may be relevant for further exploration:
{related_tools_data}

Your task:
- Answer the user's request clearly and directly, using the provided data.
- Do not mention internal tool names, file names, or implementation details.
- Present the information as if you are an expert advisor, not a software system.
- If there are multiple relevant data points, summarize them in a way that best addresses the user's question.
- If appropriate, round down decimal values for readability, but do not change the units.
- At the end, suggest one or more of the related analyses as a possible next step for the user, phrased in a friendly and helpful way.

Be concise, helpful, and approachable.
""")

async def generate_answer(state):
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

    tool_data, layers = reduce(
        lambda res, d: (
            res[0] + f"['description': {toolbox.get(d[0]).description}, 'value': {d[1][1]}]" + "\n",
            res[1] + [d[1][0]] if d[1][0] != "" else []
        ),
        state.geocontext.context.items(),
        ("", [])
    )

    # retrieve similar tools
    # to add them to the prompt
    # and lead conversation to
    # similar/related datasources
    tools = toolbox.get_last_retrieved_tools()
    # related_tools_data = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) if tools else None
    related_tools_data = "\n".join([f"- {tool.name.replace('_', ' ')}" for tool in tools]) if tools else None

    writer({"type": "info", "content": "Organizing the information."})
    prompt = answer_prompt.format(
        location=state.router.location,
        tool_data=tool_data,
        aggregated_query=state.router.aggregated_query,
        related_tools_data=related_tools_data
    )
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
        "messages": state.messages + [AIMessage(content=response.content)]
    }
