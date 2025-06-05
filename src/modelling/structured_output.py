"""This file exports Pydantic models for data used throughout the conversation context."""

from typing import Optional, Any
from pydantic import BaseModel, Field

class RouterOutput(BaseModel):
    """Router output used to route user queries to appropriate agents and retrieve basic context."""

    intent: Optional[str] = Field(
        description="Specifies the type of query based on user intent. Must be one of: 'factual' (if the user is directly asking for facts, data, or statistics), or 'actionable' (if the user is inquiring about actions, measures, or evaluations that can be taken or considered for the specified topic). If it is difficult to distinguish, assume the user is inquiring about 'factual'.",
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
        description=(
            """Set to True if you need more information to understand what the user wants (missing location, unclear intent, or vague request).
            Set to False if the request is clear and you understand what the user is asking for."""
        ),
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

    memory: str = Field()
    context: str = Field()
