from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.func import task

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from modelling.structured_output import RouterOutput, GeoContextOutput
from modelling.utils import reduce_missing_attributes

from provider.GeoSessionProvider import GeoSessionProvider

from typing import Optional, Any
from pydantic import BaseModel, Field, ValidationError

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

llm = ChatOllama(model="llama3.2:3b", temperature=0)
parser = PydanticStreamOutputParser(pydantic_object=GeoContextOutput, diff=True)

async def geocontext_retriever(state):
    """
    Retrieves relevant geographical and contextual data based on the user query.

    This function:
      - Extracts the last human message from the state.
      - Formats a routing prompt using a predefined template.
      - Invokes a language model to classify and summarize the user's query.
      - Retrieves relevant geographic and contextual data based on the classified intent.
      - Augments the conversation state with this retrieved contextual information.
      - Handles retrieval errors and determines if additional data sources are needed.

    Args:
        state: The current conversation state to which we add the retrieved geo-context.

    Returns:
        dict: The updated conversation state with.
    """

    last_human_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))

    context = GeoContextOutput()

    router_state: RouterOutput = state.router
    try:
        # instantiate potentially needed
        # geometry sessions and schemas
        # based on router location
        if router_state.location is not None:
            provider = GeoSessionProvider.get_or_create(router_state.location, 100, 0.3)
            GeoSessionProvider.get_or_create(router_state.location, 100, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 500, 1.0)
            GeoSessionProvider.get_or_create(router_state.location, 1000, 1.0)

            # await and fill context
            # with SFSO number of
            # municipality
            await provider.wait_until_sfso_ready()
            context.municipality_sfso_number = provider.municipality_sfso_number
        else:
            raise Exception("No location provided in router_state.")
    except Exception as e:
        print(f"Exception: {e}")
        return state

    # INSTANTIATE TOOLBOX

    # prompt: str = router_prompt.format(
    #     format_description=parser.get_description(),
    #     format_instructions=parser.get_format_instructions(),
    #     user_input=last_human_message
    # )
    # # invoke llm on user query
    # response = await llm.ainvoke(prompt)

    # return {}
