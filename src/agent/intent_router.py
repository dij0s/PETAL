from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser

from typing import Optional
from pydantic import BaseModel, Field, ValidationError

router_prompt = PromptTemplate.from_template("""
You are an AI assistant helping to route user requests about energy planning in Switzerland.
Classify the user input into:
- intent: one of ["data_request", "planning_request", "policy_question"]
- topic: e.g. "solar", "biomass", "heating", "wind"
- location: if available (e.g. a municipality name)

Return ONLY the following JSON like this, with no extra text, explanation, or formatting:

{{
"intent": "...",
"topic": "...",
"location": "..."
}}

User input: "{user_input}"
""")

class RouterOutput(BaseModel):
    """Router output used to route user queries to appropriate agents and retrieve basic context"""

    intent: Optional[str] = Field(description="The intent of the user, must be one of ['data_request', 'planning_request', 'policy_question']")
    topic: Optional[str] = Field(description="The main topic of the user request, e.g. 'solar', 'biomass', 'heating', 'wind'")
    location: Optional[str] = Field(description="The location mentioned in the user request, if available (e.g. a municipality name)")

llm = ChatOllama(model="llama3.2:3b", temperature=0)
parser = PydanticStreamOutputParser(pydantic_object=RouterOutput, diff=True)

async def router_llm_node(state):
    last_human_message = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    prompt: str = router_prompt.format(user_input=last_human_message) + parser.get_format_instructions()

    output = await llm.ainvoke(prompt)
    content = output.content

    try:
        parsed = parser.parse(output.content)
        print(parsed)
        if parsed is None:
            raise Exception("No parsed output")
    except Exception as e:
        print(f"Error: {e}")
        parsed = RouterOutput(intent=None, topic=None, location=None)

    return {**state, **parsed.dict()}
