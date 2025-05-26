from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from modelling.utils import reduce_missing_attributes
from provider.ToolProvider import ToolProvider

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
    {tool_description_data}

User request: "{aggregated_query}"

Respond to the user's request by utilizing the provided data in a relevant manner. You may round down decimal values if needed BUT DON'T MODIFY THE UNITS.
Be friendly and offer one specific way you can assist the user, making sure your suggestion directly relates to their request.
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

    tool_description_data = "\n".join([
        f"['description': {toolbox.get(k).description}, 'value': {v[1]}]"
        for k, v in state.geocontext.context.items()
    ])
    prompt = answer_prompt.format(location=state.router.location, tool_description_data=tool_description_data, aggregated_query=state.router.aggregated_query)

    response = await llm.ainvoke(prompt)
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)]
    }
