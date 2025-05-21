from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.types import interrupt

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from modelling.structured_output import RouterOutput
from modelling.utils import construct_clarification_prompt

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

llm = ChatOllama(model="llama3.2:3b", temperature=0).with_structured_output(RouterOutput)
parser = PydanticStreamOutputParser(pydantic_object=RouterOutput, diff=True)

async def intent_router(state):
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
    current: RouterOutput = RouterOutput(intent=None, topic=None, location=None, needs_clarification=None)
    if state.router is not None:
        current = state.router

    # assume state, by default, is Pydantic
    parsed: RouterOutput | Any = response
    try:
        if isinstance(response, dict) and hasattr(response, "content"):
            parsed = parser.parse(response.content)
            if parsed is None:
                raise Exception("No parsed output")
    except Exception as e:
        print(f"Error: {e}")
        parsed = RouterOutput(intent=None, topic=None, location=None, needs_clarification=True)

    # evaluate if state is sufficient
    # for further handing off
    if parsed.needs_clarification:
        # create user returned
        # interrupt prompt
        clarification_prompt = construct_clarification_prompt(parsed)
        # interrupt(clarification_prompt)
        print("User query should be clarified")

    # new state does not need further
    # clarification, update overall state
    updated_state = current.model_dump()
    parsed_state = parsed.model_dump()
    for k, v in updated_state.items():
        new_v = parsed_state.get(k, None)
        if (new_v is not None) and (new_v != v):
            print(f"UPDATING CURRENT KEY {k} FROM {v} TO {new_v}")
            updated_state[k] = new_v

    # update graph state
    return {**state.dict(), "router": RouterOutput(**updated_state)}
