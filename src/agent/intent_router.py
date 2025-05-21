from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser

from typing import Optional
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

# define exptected data scheme
class RouterOutput(BaseModel):
    """Router output used to route user queries to appropriate agents and retrieve basic context"""

    intent: Optional[str] = Field(description="The intent of the user, must be one of ['data_request', 'planning_request', 'policy_question']")
    topic: Optional[str] = Field(description="The main topic of the user request, e.g. 'solar', 'biomass', 'heating', 'wind'")
    location: Optional[str] = Field(description="The location mentioned in the user request, if available (e.g. a municipality name)")

llm = ChatOllama(model="llama3.2:3b", temperature=0).with_structured_output(RouterOutput)
parser = PydanticStreamOutputParser(pydantic_object=RouterOutput, diff=True)

async def router_llm_node(state):
    last_human_message = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    prompt: str = router_prompt.format(
        format_description=parser.get_description(),
        format_instructions=parser.get_format_instructions(),
        user_input=last_human_message
    )
    # invoke llm on user query
    response = await llm.ainvoke(prompt)
    # parse response accordingly to
    # enable further actions based
    # on predefined scheme
    if isinstance(response, RouterOutput):
        print(response)
        return {**state, **response.dict()}
    else:
        try:
            if isinstance(response, dict) and hasattr(response, "content"):
                parsed = parser.parse(response.content)

                if parsed is None:
                    raise Exception("No parsed output")

                return {**state, **parsed.dict()}
        except Exception as e:
            print(f"Error: {e}")
            parsed = RouterOutput(intent=None, topic=None, location=None)

            return {**state, **parsed.dict()}
