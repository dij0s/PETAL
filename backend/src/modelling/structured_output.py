"""This file exports Pydantic models for data used throughout the conversation context."""

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from typing import Optional, Any, Annotated
from pydantic import BaseModel, Field

class RouterOutput(BaseModel):
    """Router output used to route user queries to appropriate agents and retrieve basic context."""

    intent: Optional[str] = Field(
        description="Specifies the type of query based on user intent. Must be one of: 'factual' (user is requesting specific data points, statistics, or current state information without seeking planning guidance), or 'actionable' (user is seeking planning guidance, strategic evaluation, or implementation advice, including questions about importance, potential, value, recommendations, strategies, evaluation, implementation, opportunities, or suitability). When in doubt, if the question could benefit from regulatory guidelines, strategic context, or planning methodology, classify as 'actionable'.",
        default=None
    )
    location: Optional[str] = Field(
        description="The location mentioned in the user request, if available (ONLY SUPPORTS THE municipality name)", default=None
    )
    aggregated_query: Optional[str] = Field(
        description="An aggregated summary of the user request, combining all available context from the conversation, including follow-up exchanges. Summarize in a way that merges the relevant turns, without adding extra or hallucinated information ensuring that all content is appropriately translated to English as needed.",
        default=None
    )
    conversation_type: str = Field(
       description="Identifies the conversational context to determine appropriate response format. Must be one of: 'new_analysis' (fresh query requiring comprehensive structured response with full data analysis framework), 'correction_request' (user questioning accuracy of previous response, using phrases like 'are you sure', 'that seems wrong', 'incorrect', or pointing out specific errors that need direct acknowledgment), 'follow_up' (user requesting additional detail or expansion on the same topic from previous response). When in doubt between correction_request and other types, look for explicit doubt about accuracy or specific figure questioning.",
       default="new_analysis"
    )
    needs_clarification: bool = Field(
        description="""Set to True if you need more information to understand what the user wants (missing location, unclear intent, or vague request). Set to False if the request is clear and you understand what the user is asking for.""",
        default=True
    )
    needs_memoization: bool = Field(
        description="""Set to True ONLY when the user provides explicit preferences, corrections to assumptions, or scope refinements that should be remembered for future queries. Examples: user specifies they only want electricity data (not total energy), corrects interpretation of technical terms, or establishes recurring analysis preferences. Focus on learning user preferences that improve future responses.""",
        default=False
    )

class GeoContextOutput(BaseModel):
    """GeoContext Retriever output used to fetch relevant data from the user query and process it further."""

    context_tools: dict[str, tuple[str, Any, str, str]] = Field(
        description="Maps tool called to the retrieved layer, data, short description and source",
        default_factory=dict
    )
    context_constraints: list[tuple[str, str]] = Field(
        description="A list of constraints, each as a tuple containing the constraint content and its source.",
        default_factory=list
    )

class CriticOutput(BaseModel):
    retry: bool = Field(description="Whether the prompt should be retried against the pipeline", default=False)

class Memory(BaseModel):
    """Memory schema definition."""
    memory: str
    context: str
    timestamp: float

class Stats(BaseModel):
    """User statistics schema definition"""
    token_usage_mean: float
    token_usage_M2: float
    chat_calls_count: int
    timestamp: float

class StatsPatch(BaseModel):
    """Single run user statistics"""
    token_usage: Optional[int] = None

    def reduce(self, other: "StatsPatch") -> "StatsPatch":
        return StatsPatch(token_usage=(self.token_usage or 0) + (other.token_usage or 0))

class BenchmarkScore(BaseModel):
    data_interpretation: int = Field(description="Score from 1 to 5 indicating how accurately the response interprets and presents data.")
    guideline_application: int = Field(description="Score from 1 to 5 indicating how well the response applies cantonal guidelines to municipal planning.")
    municipal_relevance: int = Field(description="Score from 1 to 5 indicating how relevant and actionable the response is for municipal energy planning.")
    source_citations: int = Field(description="Score from 1 to 5 indicating the quality and accuracy of source citations.")
    specific_issues: list[str] = Field(description="List of any specific problems found across all criteria.", default_factory=list)

def _geocontext_reducer(a: GeoContextOutput | dict, b: GeoContextOutput | dict) -> GeoContextOutput:
    """
    Custom reducer for concurrent GeoContextOutput state updates.

    Args:
        a (GeoContextOutput): The first GeoContextOutput instance.
        b (GeoContextOutput): The second GeoContextOutput instance.

    Returns:
        GeoContextOutput: The reduced GeoContextOutput instance combining information from both inputs.
    """
    # reduce to state whose
    # individual keys are the
    # one with the most data
    updated_state = GeoContextOutput()
    if isinstance(a, dict):
        a = GeoContextOutput(**a)
    if isinstance(b, dict):
        b = GeoContextOutput(**b)
    # if geocontext is cleared,
    # we must properly overwrite
    # the current geocontext
    if (a == b) or (b == GeoContextOutput()):
        return b

    if len(a.context_tools) >= len(b.context_tools):
        updated_state.context_tools = a.context_tools
        if len(a.context_constraints) > len(b.context_constraints):
            updated_state.context_constraints = a.context_constraints
        else:
            updated_state.context_constraints = b.context_constraints
    else:
        # per definition as each state
        # is only updated by a single
        # node we can deduce the other
        # from evaluating the first one
        updated_state.context_tools = b.context_tools
        updated_state.context_constraints = a.context_constraints

    return updated_state

def _override_reducer(_, b):
    # b being the new state
    return b

class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    # router is ensured to be the same at every superstep
    router: Annotated[Optional[RouterOutput], _override_reducer] = None
    geocontext: Annotated[Optional[GeoContextOutput], _geocontext_reducer] = None
    # lang is ensured to be the same at every superstep
    lang: Annotated[str, _override_reducer] = "en"

class PromptRequest(BaseModel):
    user_id: str
    thread_id: str
    prompt: str
    lang: Optional[str] = None
    checkpoint_data: Optional[State] = None
