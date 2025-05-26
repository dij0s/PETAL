from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama

from provider.ToolProvider import ToolProvider

from functools import reduce

llm = ChatOllama(model="llama3.2:3b", temperature=0.8)

answer_prompt = PromptTemplate.from_template("""
You are an AI assistant answering a user request about energy planning in Switzerland.

The data for the user's requested location has already been gathered and is available for your use.
It is reported in this format :
    [
        "description": "Describes the format of the data and what it represents. Includes the units.
        "value": "The associated data"
    ]
    ...

Gathered available data for {location}:
    {tool_data}

User request: "{aggregated_query}"

Respond to the user's request by utilizing the provided data in a relevant manner. YOU MUST ROUND DOWN DECIMAL VALUES TO A MAXIMUM OF 2 DIGITS BUT DON'T MODIFY THE UNITS.

Offer one specific way you can assist the user, making sure your suggestion directly relates to their request. You can easily retrieve the following data. DO NOT REVEAL THE FUNCTION NAME OR ANYTHING. Just use the description to guide you :

{available_tools}
""")

async def generate_answer(state):
    """
    Generates an appropriate answer to the user's request.

    Args:
        state: The current conversation state

    Returns:
        A dictionary with updated messages including the generated answer
    """
    # retrieve description of
    # aggregated data using tools
    toolbox: ToolProvider = await ToolProvider.acreate(state.router.location)

    tool_data, tool_layers = reduce(
        lambda res, d: (
            res[0] + f"['description': {toolbox.get(d[0]).description}, 'value': {d[1][1]}]" + "\n",
            res[1] + [d[1][0]]
        ),
        state.geocontext.context.items(),
        ("", [])
    )

    # retrieve similar tools
    # to add them to the prompt
    # and lead conversation to
    # similar/related datasources
    tools = toolbox.get_last_retrieved_tools()
    available_tools = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) if tools else None

    prompt = answer_prompt.format(
        location=state.router.location,
        tool_data=tool_data,
        aggregated_query=state.router.aggregated_query,
        available_tools=available_tools
    )

    response = await llm.ainvoke(prompt)
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)]
    }
