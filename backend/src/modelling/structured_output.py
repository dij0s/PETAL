"""This file exports Pydantic models for data used throughout the conversation context."""

from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages

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
    needs_clarification: bool = Field(
        description="""Set to True if you need more information to understand what the user wants (missing location, unclear intent, or vague request). Set to False if the request is clear and you understand what the user is asking for.""",
        default=True
    )
    needs_memoization: bool = Field(
        description="""Set to True if the user corrects, clarifies, or expresses dissatisfaction with a previous answer. Look for words like "No", "Actually", "I meant", or when the user provides a correction. Otherwise, set to False.
        """,
        default=False
    )

class GeoContextOutput(BaseModel):
    """GeoContext Retriever output used to fetch relevant data from the user query and process it further."""

    context_tools: dict[str, tuple[str, Any]] = Field(
        description="Maps tool called to the retrieved layer and data of any type",
        default_factory=dict
    )
    context_constraints: list[tuple[str, str]] = Field(
        description="A list of constraints, each as a tuple containing the constraint content and its source.",
        default_factory=list
    )

class Memory(BaseModel):
    """Memory schema definition."""
    memory: str
    context: str
    timestamp: float

class BenchmarkScore(BaseModel):
    data_interpretation: int = Field(description="Score from 1 to 5 indicating how accurately the response interprets and presents data.")
    guideline_application: int = Field(description="Score from 1 to 5 indicating how well the response applies cantonal guidelines to municipal planning.")
    municipal_relevance: int = Field(description="Score from 1 to 5 indicating how relevant and actionable the response is for municipal energy planning.")
    source_citations: int = Field(description="Score from 1 to 5 indicating the quality and accuracy of source citations.")
    specific_issues: list[str] = Field(description="List of any specific problems found across all criteria.", default_factory=list)

class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    router: Optional[RouterOutput] = None
    geocontext: Optional[GeoContextOutput] = None
    lang: str = "en"

class PromptRequest(BaseModel):
    user_id: str
    thread_id: str
    prompt: str
    lang: Optional[str] = None
    checkpoint_data: Optional[State] = None
