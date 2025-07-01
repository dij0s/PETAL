from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.exceptions import OutputParserException
from langgraph.store.base import BaseStore
from langchain_core.callbacks import AsyncCallbackManager
from langgraph.config import get_stream_writer

from provider.ModelProvider import ModelProvider
from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider
from provider.callbacks import CustomCallback
from storage.user import fetch_stats
from modelling.utils import bin
from modelling.structured_output import State, CriticScore, CriticOutput, Stats

from functools import reduce

llm = (
    ModelProvider
        .from_env_variable(
            env_variable="OLLAMA_MODEL_LLM_ANSWERING",
            temperature=0.8,
            defaults="qwen3:1.7b",
        )
).with_structured_output(CriticScore)

system_prompt = PromptTemplate.from_template("""
Given that the municipality "{location}" has {residents_count} residents and an exploitable surface of {exploitable_surface} ha, we redacted the following report to answer the user request.

This is the data we have at our disposal:

### Datapoints for "{location}"
{datapoints_description}

And this is our answer to the user request "{user_request}":

{llm_answer}

Only check for common interpretation errors:
1. Unit confusion (GWh vs MWh vs kWh)
2. Per-capita vs total confusion
3. Annual vs instantaneous values
4. Potential vs actual production mixing

Please provide in output a single score between 0 and 1.
""")

async def critic_answer(state: State, *, config: RunnableConfig, store: BaseStore) -> CriticOutput:
    """
    Evaluates and critiques the answer given to the user's request.

    Args:
        state (State): The current conversation state
        config (RunnableConfig): The configuration for the runnable.
        store (BaseStore): The long-term memory store.

    Returns:
        CriticOutput: The private state indicating if the prompt should be retried.
    """
    writer = get_stream_writer()
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

    # retrieve contextual data
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

    try:
        await provider.wait_until_residents_count_ready()
        prompt = [
            SystemMessage(content=system_prompt.format(
                location=state.router.location,
                residents_count=provider.residents_count,
                exploitable_surface=f"{provider.exploitable_surface:.2f}",
                datapoints_description=datapoints_description,
                user_request=last_human_message,
                llm_answer=last_ai_message,
            ))
        ]

        response = await llm.ainvoke(prompt)
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        # no retry as this is probably
        # due to external factors which
        # mean residents count is not
        # retrieveable
        return CriticOutput(retry=False)
    except OutputParserException as e:
        print(f"Could not parse output into Pydantic definition: {e}")
        writer({"type": "retry", "content": "Not satisfied with the answer. Let's retry."})
        return CriticOutput(retry=True)

    # bin and push greenness
    # indicator from current
    # statistics
    try:
        current_stats = await fetch_stats(config, store)
        if current_stats is not None:
            manager = config.get("callbacks")
            if not isinstance(manager, AsyncCallbackManager):
                raise ValueError("Invalid callback manager")

            callback = next((handler for handler in manager.handlers if isinstance(handler, CustomCallback)), None)
            if callback is None:
                raise ValueError("No CustomCallback handler found")

            previous_stats = callback.last_run_state
            if previous_stats is not None:
                # bin score into single score
                # and push to frontend
                score = bin(previous_stats, current_stats)
                writer({"type": "greenness", "content": score})
    except Exception as e:
        print(f"Exception: {e}")

    if isinstance(response, CriticScore):
        # only retry if scores is low
        # and there are clear issues
        output = CriticOutput(retry=((response.score < 0.6) and (len(response.issues) > 0)))
        if output.retry:
            writer({"type": "retry", "content": "Not satisfied with the answer. Let's retry."})

        return output

    writer({"type": "retry", "content": "Not satisfied with the answer. Let's retry."})
    return CriticOutput(retry=True)
