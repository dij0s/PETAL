"""This file exports Pydantic models for data used throughout the conversation context."""

from typing import Optional, Any
from pydantic import BaseModel, Field

class RouterOutput(BaseModel):
    """Router output used to route user queries to appropriate agents and retrieve basic context."""

    intent: Optional[str] = Field(
        description="The user's underlying goal or type of request. Must be one of: 'data_request' (seeking factual data or statistics), 'planning_request' (asking for help with planning, scenarios, or recommendations), or 'policy_question' (inquiring about laws, regulations, or policy matters).",
        default=None
    )
    topic: Optional[str] = Field(description="The main topic of the user request, e.g. 'solar', 'biomass', 'heating', 'wind', if available.", default=None)
    location: Optional[str] = Field(description="The location mentioned in the user request, if available (e.g. a municipality name)", default=None)
    needs_clarification: bool = Field(description="Indicates whether the user's request is unclear or incomplete and requires additional information or clarification before it can be properly routed or answered.", default=True)

class GeoContextOutput(BaseModel):
    """GeoContext Retriever output used to fetch relevant data from the user query and process it further."""

    municipality_sfso_number: Optional[int] = Field(description="The official numerical identifier assigned by the Swiss Federal Statistical Office to the municipality", default=None)
    context: dict[str, tuple[str, Any]] = Field(description="Maps tool called to the retrieved layer and data of any type", default_factory=dict)
