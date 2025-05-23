from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from modelling.utils import reduce_missing_attributes

llm = ChatOllama(model="llama3.2:3b", temperature=0.95)

clarification_prompt = PromptTemplate.from_template("""
The user just queried some information and you need additional details about:

{needed_information}

Formulate a question asking for these specific details.
If there is no extra needed information, then, they must have mistakenly input something.
Keep the answer very CONCISE and address the user directly.
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
    missing_attributes = reduce_missing_attributes(state.router)
    prompt = clarification_prompt.format(needed_information=missing_attributes)
    print(prompt)
    response = await llm.ainvoke(prompt)

    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)]
    }
