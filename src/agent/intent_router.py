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

    Here is the previous conversation context that you have already deduced from earlier user inputs and THE LAST USER INPUT. Use this information to help clarify the current request:
    Previous user input: "{previous_user_input}"
    Conversation context: "{current_router}"

    User input: "{user_input}"
    """)

llm = ChatOllama(model="llama3.2:3b", temperature=0).with_structured_output(RouterOutput)
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

    human_messages = [msg.content for msg in state.messages if isinstance(msg, HumanMessage)]
    last_human_message = human_messages[-1] if human_messages else ""
    previous_human_message = human_messages[-2] if len(human_messages) > 1 else ""

    # retrieve curent router context
    # and fill it in the prompt for
    # context carry over
    current: RouterOutput = RouterOutput(intent=None, topic=None, location=None, needs_clarification=True)
    if state.router is not None:
        current = state.router

    prompt: str = router_prompt.format(
        current_router=current.model_dump(),
        format_description=parser.get_description(),
        format_instructions=parser.get_format_instructions(),
        previous_user_input=previous_human_message,
        user_input=last_human_message
    )
    # invoke llm on user query
    response = await llm.ainvoke(prompt)
    # parse response accordingly to
    # enable further actions based
    # on predefined scheme ; first
    # try to get last state to only
    # update it

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
    # update graph state
    # not destructuring the messages
    # using the dot notation on the
    # State instance serializes the
    # message itself as we called
    # the _dict_ function on the State
    return {"messages": state.messages + [AIMessage(content="Let me process your query...")], "router": RouterOutput(**updated_state)}
