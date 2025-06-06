import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore
from langgraph.config import get_stream_writer

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from modelling.structured_output import GeoContextOutput, RouterOutput, Memory
from modelling.utils import reduce_missing_attributes
from storage.memories import fetch_memories, update_memories

from typing import Optional, Any
from pydantic import BaseModel, Field, ValidationError

from functools import reduce

# define system prompt for enhanced
# formatting and data scheme validation
system_prompt = PromptTemplate.from_template("""
You are an AI assistant helping to route user requests about energy planning in Switzerland.

Your task is to extract and update metadata from user queries to maintain conversation context.

## Classification Schema:
{format_description_llm}

## Key Instructions for aggregated_query:
- **ALWAYS update aggregated_query** with the current user's request
- If this is a new topic/question: Create a fresh, comprehensive query
- If this is a follow-up/clarification: Merge the previous context with the new information
- If this is a correction: Replace the incorrect parts with the corrected information
- Keep the aggregated query focused and specific - don't add assumptions

## Examples:

**New Question:**
User: "What are the energy needs for households in Sion?"
→ aggregated_query: "Energy needs for households in Sion"

**Follow-up Clarification:**
Previous: "Energy needs for households in Sion"
User: "Thanks a lot but when I ask for the energy needs for households, I only mean the electricity needs."
→ aggregated_query: "Electricity needs for households in Sion"
→ needs_memoization: True

**Scope Correction:**
Previous: "Energy consumption in Lausanne"
User: "No, I meant for my apartment building, not the whole city."
→ aggregated_query: "Energy consumption for apartment buildings in Lausanne"
→ needs_memoization: True

**Topic Expansion:**
Previous: "Solar energy potential in Geneva"
User: "What about wind energy too?"
→ aggregated_query: "Solar and wind energy potential in Geneva"
→ needs_memoization: False

**Completely New Topic:**
Previous: "Heating costs in Zurich"
User: "What can you tell me about Martigny?"
→ aggregated_query: "General information about Martigny"
→ needs_memoization: False
""")

user_prompt = PromptTemplate.from_template("""
**Previous Context:**
- Last input: "{previous_user_input}"
- Current state: {current_router}

**Current Input:** "{user_input}"
""")

MODEL = os.getenv("OLLAMA_MODEL_LLM_ROUTING", "llama3.2:3b")
llm = ChatOllama(model=MODEL, temperature=0).with_structured_output(RouterOutput, method="function_calling")
parser = PydanticStreamOutputParser(pydantic_object=RouterOutput, diff=True)

async def intent_router(state, *, config: RunnableConfig, store: BaseStore):
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
    writer = get_stream_writer()
    # retrieve messages for prompt
    human_messages: list[str] = [msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage)] # type: ignore
    last_human_message: str = human_messages[0] if human_messages else ""
    previous_human_message: str = human_messages[1] if len(human_messages) > 1 else ""
    # retrieve curent router context
    # and fill it in the prompt for
    # context carry over
    current: RouterOutput = RouterOutput(intent=None, location=None, needs_clarification=True)
    if state.router is not None:
        current = state.router

    prompt = [
        SystemMessage(content=system_prompt.format(
            format_description_llm=parser.get_description(),
        )),
        HumanMessage(content=user_prompt.format(
            current_router=current.model_dump(),
            previous_user_input=previous_human_message,
            user_input=last_human_message,
        ))
    ]
    # write custom event
    writer({"type": "info", "content": "Interpreting your request..."})
    # invoke llm on prompt
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
        parsed = RouterOutput(intent=None, location=None, needs_clarification=True)

    # new state does not need further
    # clarification, update overall state
    # buffered last location to know if
    # geocontext reset is appropriate
    last_location = current.location
    updated_state = current.model_dump()
    parsed_state = parsed.model_dump()
    for k, v in updated_state.items():
        new_v = parsed_state.get(k, None)
        if new_v not in (None, "null", "", v) and (new_v != v):
            updated_state[k] = new_v

    # explicitly set the flag for extra
    # clarification to False if all the
    # context fields are set
    if updated_state["needs_clarification"]:
        all_fields_set = all([
            v is not None
            for k, v in updated_state.items()
            if k not in ["needs_clarification", "needs_memoization"]
        ])
        updated_state["needs_clarification"] = not all_fields_set


    # memoize information if needed
    # and toggle flag back to False
    # as this is only used in the
    # business logic itself
    if updated_state["needs_memoization"]:
        await update_memories(config, store, last_human_message, previous_human_message)
        writer({"type": "info", "content": "I'll know that the next time!"})

        updated_state["needs_memoization"] = False

    updated_router = RouterOutput(**updated_state)
    print(updated_router)
    # reset geocontext on location change
    if last_location != updated_router.location:
        return {
            **state.model_dump(),
            "messages": state.messages,
            "router": updated_router,
            "geocontext": GeoContextOutput(),
        }
    # update graph state
    # not destructuring the messages
    # using the dot notation on the
    # State instance serializes the
    # message itself as we called
    # the _dict_ function on the State
    return {
        **state.model_dump(),
        "messages": state.messages,
        "router": updated_router,
    }
