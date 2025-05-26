"""This file exports Pydantic models for data used throughout the conversation context."""

from typing import Optional, Any
from pydantic import BaseModel, Field

class RouterOutput(BaseModel):
    """Router output used to route user queries to appropriate agents and retrieve basic context."""

    intent: Optional[str] = Field(
    description="The user's underlying goal or type of request. Must be one of: 'data' (seeking factual data or statistics), 'planning' (asking for help with planning, scenarios, or recommendations), or 'policy' (inquiring about laws, regulations, or policy matters). If it is difficult to distinguish, assume the user is inquiring about data.",
        default=None
    )
    # data_type: Optional[str] = Field(
    #     description=(
    #         "Specifies the type of data the user is inquiring about. Must be one of: 'measures' (precise measured values, e.g., heating needs, energy consumption), 'potential' (estimates of possible gains or capacity, e.g., how much could be produced or saved), or 'infrastructure' (counts or details of physical assets, e.g., number of wind turbines, thermal networks, hydroelectric plants)."
    #     ),
    #     default=None
    # )
    topic: Optional[str] = Field(description="The main topic of the user request, e.g. 'solar', 'biomass', 'heating', 'wind', if available.", default=None)
    location: Optional[str] = Field(description="The location mentioned in the user request, if available (ONLY SUPPORTS THE municipality name)", default=None)
    aggregated_query: Optional[str] = Field(
        description="A natural language summary of the complete user request, synthesized from all available context. As short as possible while being as descriptive as possible WITHOUT EXTRA HALLUCINATED INFORMATION.",
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

    # municipality_sfso_number: Optional[int] = Field(description="The official numerical identifier assigned by the Swiss Federal Statistical Office to the municipality", default=None)
    context: dict[str, tuple[str, Any]] = Field(description="Maps tool called to the retrieved layer and data of any type", default_factory=dict)
