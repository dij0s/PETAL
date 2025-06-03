import os

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer

from provider.GeoSessionProvider import GeoSessionProvider
from provider.ToolProvider import ToolProvider

from collections import defaultdict
from functools import reduce

MODEL = os.getenv("OLLAMA_MODEL_LLM", "llama3.2:3b")
llm = ChatOllama(model=MODEL, temperature=0.8)

full_language: defaultdict[str, str] = defaultdict(lambda: "English", {
    "fr": "French",
    "de": "German",
})

system_prompt = PromptTemplate.from_template("""
You are an AI assistant specializing in energy planning for {location}.

Your task:
- Answer the user's request only using the provided data.
- If you don't know the answer to the user's request, just say it.
- If there are multiple relevant data points, summarize them in a way that best addresses the user's question.
- Present the information as if you are an expert advisor, not a software system.
- Do not mention internal tool names, file names, or implementation details.
- If appropriate, round down decimal values for readability, but do not change the units. ALWAYS INCLUDE UNITS IN THE ANSWER.
- At the end, suggest one or more of the related analyses as a possible next step for the user, phrased in a friendly and helpful way.
Be concise, helpful, and approachable.
Please respond in {lang}, ensuring that all content is appropriately translated as needed.
""")

user_prompt_with_constraints = PromptTemplate.from_template("""
You have already gathered the relevant data for the location "{location}". This data is provided below, with each entry including a description (explaining what the data represents and its units) and the corresponding value.

Available data:
{tools_data}

User request: "{aggregated_query}"

The legislation and other relevant documents for effective energy planning provided us with additional information on the matter. Make sure to include this information, as these are the guidelines from the state and country. Explain the goals and constraints in detail, without omitting any possible dates mentioned:
{constraints}

Here are some related data sources that may be relevant for the user in the same categorie(s) "{categories}":
{related_tools_description}
""")

user_prompt_no_constraints = PromptTemplate.from_template("""
You have already gathered the relevant data for the location "{location}". This data is provided below, with each entry including a description (explaining what the data represents and its units) and the corresponding value.

Available data:
{tools_data}

User request: "{aggregated_query}"

Here are some related data sources that may be relevant for the user in the same categorie(s) "{categories}":
{related_tools_description}
""")

async def generate_answer(state):
    """
    Generates an appropriate answer to the user's request.

    Args:
        state: The current conversation state

    Returns:
        A dictionary with updated messages including the generated answer
    """
    writer = get_stream_writer()
    provider = GeoSessionProvider.get_or_create(state.router.location, 100, 0.3)
    # retrieve description of
    # aggregated data using tools
    toolbox: ToolProvider = await ToolProvider.acreate(state.router.location)
    tools_data, layers = reduce(
        lambda res, d: (
            res[0] + f"['description': {toolbox.get(d[0]).description}, 'value': {d[1][1]}]" + "\n",
            res[1] + [d[1][0]] if d[1][0] != "" else []
        ),
        state.geocontext.context_tools.items(),
        ("", [])
    )
    # retrieve similar tools
    # from same category to
    # better lead conversation
    last_categories = toolbox.get_last_retrieved_categories()
    if last_categories is None:
        last_categories = []
    related_tools = reduce(
        lambda res, c: [*res, *toolbox.get_tools(c)],
        last_categories,
        []
    )
    # don't consider actual tools
    # which we've already fetched
    related_tools = [tool for tool in related_tools if tool.name not in state.geocontext.context_tools.keys()]
    related_tools_description = "\n".join(tool.description for tool in related_tools)

    writer({"type": "info", "content": "Organizing the information."})
    writer({"type": "log", "content": f"Providing {state.router.intent} information, constraining context is of length {len(state.geocontext.context_constraints)}"})
    # build prompt based on factual
    # or actionable user request
    prompt_args = {
        "location": state.router.location,
        "tools_data": tools_data,
        "aggregated_query": state.router.aggregated_query,
        "categories": last_categories,
        "related_tools_description": related_tools_description,
        "lang": full_language.get(state.lang),
    }
    if state.router.intent != "factual":
        prompt_args["constraints"] = state.geocontext.context_constraints
        user_prompt = user_prompt_with_constraints
    else:
        user_prompt = user_prompt_no_constraints

    prompt = [
        SystemMessage(content=system_prompt.format(**prompt_args)),
        HumanMessage(content=user_prompt.format(**prompt_args))
    ]
    response = await llm.ainvoke(prompt)

    # update state with response
    # and push the new layers and
    # municipality's SFSO number
    # if there are any layers
    if len(layers) > 0:
        await provider.wait_until_sfso_ready()
        writer({"type": "layers", "layers": layers})
        writer({"type": "sfso_number", "sfso_number": provider.municipality_sfso_number})

    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=response.content)]
    }
