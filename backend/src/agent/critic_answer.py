from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from provider.ModelProvider import ModelProvider
from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider
from modelling.structured_output import State

from functools import reduce

llm = (
    ModelProvider
        .from_env_variable(
            env_variable="OLLAMA_MODEL_LLM_ANSWERING",
            temperature=0.8,
            defaults="qwen3:1.7b",
        )
)

system_prompt = PromptTemplate.from_template("""
Given that the municipality "{location}" has {residents_count} residents, a total surface of {total_surface} m² and an exploitable surface of {exploitable_surface} m², we redacted the following report to answer the user request.

This is the data we've had at our disposal:

### Datapoints for "{location}"
{datapoints_description}

And this is our answer to the user request "{user_request}":

{llm_answer}

Please provide in output a single confidence score between 0 and 1, indicating the confidence level of the answer and how probative the data looks considering the municipality's description.
""")

async def critic_answer(state: State):
    """
    Evaluates and critiques the answer given to the user's request.

    Args:
        state: The current conversation state

    Returns:
        A dictionary with updated messages including the generated answer
    """
    # ensure typesafety by
    # evaluating the state
    if state.router is None or state.geocontext is None:
        raise ValueError("State isn't properly defined")

    if state.router.location is None or state.router.aggregated_query is None:
        raise ValueError("Router state isn't properly defined")

    if state.geocontext.context_tools is None or state.geocontext.context_constraints is None:
        raise ValueError("Geocontext state isn't properly defined")

    provider: GeoSessionProvider = GeoSessionProvider.get_or_create(state.router.location, 100, 1.0, with_residents_count=True)
    toolbox: ToolProvider = await ToolProvider.acreate(state.router.location)

    # retrieve messages from the
    # last conversation chat
    last_human_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))
    last_ai_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, AIMessage))

    # format context data
    datapoints_description = reduce(
        lambda res, d: res + [f"['description': {toolbox.get(d[0]).description}, 'value': {d[1][1]}]" + "\n" if toolbox.get(d[0]) is not None else ""], # type: ignore
        reduce(
            lambda res, d: {
                **res,
                d[0]: d[1][:2]
            },
            state.geocontext.context_tools.items(),
            {}
        ).items(),
        []
    )

    prompt = [
        SystemMessage(content=system_prompt.format(
            location=state.router.location,
            residents_count=23000,
            total_surface=2000,
            exploitable_surface=200,
            datapoints_description=datapoints_description,
            user_request=last_human_message,
            llm_answer=last_ai_message,
        ))
    ]
    response = await llm.ainvoke(prompt)
    print(response.content)
    # return {
    #     "messages": [AIMessage(content=response.content)],
    # }
