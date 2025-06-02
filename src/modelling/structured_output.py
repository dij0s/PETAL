"""This file exports Pydantic models for data used throughout the conversation context."""

from typing import Optional, Any
from pydantic import BaseModel, Field

class RouterOutput(BaseModel):
    """Router output used to route user queries to appropriate agents and retrieve basic context."""

    intent: Optional[str] = Field(
        description=(
            "Specifies the type of query based on user intent. Must be one of: 'factual' (if the user is directly asking for facts, data, or statistics), or 'actionable' (if the user is inquiring about actions, measures, or evaluations that can be taken or considered for the specified topic). If it is difficult to distinguish, assume the user is inquiring about 'factual'."
        ),
        default=None
    )
    topic: Optional[str] = Field(description="The primary subject of the user's request, such as 'solar', 'biomass', 'heating', or 'wind'. **Leave empty if not specified.**", default=None)
    location: Optional[str] = Field(description="The location mentioned in the user request, if available (ONLY SUPPORTS THE municipality name)", default=None)
    aggregated_query: Optional[str] = Field(
        description="An aggregated summary of the user request, combining all available context from the conversation, including follow-up exchanges. Summarize in a way that merges the relevant turns, without adding extra or hallucinated information.",
        default=None
    )
    needs_clarification: bool = Field(
        description=(
            "Set to True if, after considering both the current user input AND THE PREVIOUS CONVERSATION CONTEXT, the question is ambiguous or if the user's request cannot be confidently routed or answered without further information. Otherwise, set to False."
        ),
        default=True
    )

class GeoContextOutput(BaseModel):
    """GeoContext Retriever output used to fetch relevant data from the user query and process it further."""

    context_tools: dict[str, tuple[str, Any]] = Field(description="Maps tool called to the retrieved layer and data of any type", default_factory=dict)
    context_constraints: list[str] = Field(description="Retrieved constraints from the described legislation and vision", default_factory=list)
