from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from modelling.structured_output import RouterOutput
from modelling.utils import reduce_missing_attributes

from typing import Optional, Any
from pydantic import BaseModel, Field, ValidationError

from functools import reduce

# define system prompt for enhanced
# formatting and data scheme validation
router_prompt = PromptTemplate.from_template("""
    You are an AI assistant helping to route user requests about energy planning in Switzerland.
    Classify the user input into:

    {format_description}

    Return ONLY the following JSON like this, with no extra text, explanation, or formatting:

    {format_instructions}

    User input: "{user_input}"
    """)

clarification_promp = PromptTemplate.from_template("""
    The user just queried some nonsense and you need additional information about this :

    {needed_information}

    If there is no extra needed information, then, they must have mistakely input something.
    Keep the answer very CONCISE and address the user directly.
    """)

llm = ChatOllama(model="llama3.2:3b", temperature=0).with_structured_output(RouterOutput)
conversation_llm = ChatOllama(model="llama3.2:3b", temperature=0.9)

parser = PydanticStreamOutputParser(pydantic_object=RouterOutput, diff=True)

async def intent_router(state):
    """
    Routes user intent based on the latest human message in the conversation state.

    This function:
      - Extracts the last human message from the state.
      - Formats a routing prompt using a predefined template.
      - Invokes a language model to classify the user input.
      - Parses the model's response into a structured RouterOutput.
      - Handles parsing errors and determines if clarification is needed.
      - Updates the routing state accordingly and returns the new state.

    Args:
        state: The current conversation state, expected to have a 'messages' attribute (list of messages)
               and an optional 'router' attribute (RouterOutput).

    Returns:
        dict: The updated conversation state with the 'router' field set to the latest RouterOutput.
    """

    last_human_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))
    prompt: str = router_prompt.format(
        format_description=parser.get_description(),
        format_instructions=parser.get_format_instructions(),
        user_input=last_human_message
    )
    # invoke llm on user query
    response = await llm.ainvoke(prompt)
    # parse response accordingly to
    # enable further actions based
    # on predefined scheme ; first
    # try to get last state to only
    # update it
    current: RouterOutput = RouterOutput(intent=None, topic=None, location=None, needs_clarification=True)
    if state.router is not None:
        current = state.router

    # assume state, by default, is Pydantic
    parsed: RouterOutput | Any = response
    try:
        if isinstance(response, dict) and "content" in response:
            parsed = parser.parse(response["content"])
            if parsed is None:
                raise Exception("No parsed output")
    except Exception as e:
        print(f"Error: {e}")
        parsed = RouterOutput(intent=None, topic=None, location=None, needs_clarification=True)

    # new state does not need further
    # clarification, update overall state
    updated_state = current.model_dump()
    parsed_state = parsed.model_dump()
    for k, v in updated_state.items():
        new_v = parsed_state.get(k, None)
        if (new_v is not None) and (new_v != v):
            updated_state[k] = new_v

    updated = RouterOutput(**updated_state)

    # only push new message to user
    # if the query needs extra clarification
    if updated.needs_clarification:
        missing_attributes = reduce_missing_attributes(updated)
        prompt = clarification_promp.format(needed_information=missing_attributes)
        response = await conversation_llm.ainvoke(prompt)

        return { **state.dict(), "messages": state.messages + [AIMessage(response.content)], "router": updated }

    # update graph state
    # not destructuring the messages
    # using the dot notation on the
    # State instance serializes the
    # message itself as we called
    # the _dict_ function on the State
    return {**state.dict(), "messages": state.messages, "router": updated}
