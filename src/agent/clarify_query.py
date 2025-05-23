from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from modelling.utils import reduce_missing_attributes

llm = ChatOllama(model="llama3.2:3b", temperature=0.95)

clarification_prompt = PromptTemplate.from_template("""
You are an AI assistant helping to clarify user requests about energy planning in Switzerland.
The user just queried some information and you need additional details about:

{needed_information}

User input: "{user_input}"

Formulate a question asking for these specific details.
If there is no extra needed information, then, they must have mistakenly input something.
Keep the answer short and address the user in a friendly, non-robotic way.
""")

async def clarify_query(state):
    """
    Creates a clarification message to ask the user for more information.

    This function analyzes the current state to identify missing information
    and generates an appropriate clarification question to the user.

    Args:
        state: The current conversation state containing messages and router info

    Returns:
        A dictionary with updated messages including the clarification
    """
    last_human_message = next(msg.content for msg in state.messages if isinstance(msg, HumanMessage))

    missing_attributes = reduce_missing_attributes(state.router)
    prompt = clarification_prompt.format(needed_information=missing_attributes, user_input=last_human_message)
    response = await llm.ainvoke(prompt)

    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)]
    }
