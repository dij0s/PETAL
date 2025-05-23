import uuid

from typing import Callable, Optional
from functools import reduce

from langchain_core.tools.structured import StructuredTool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_ollama import OllamaEmbeddings

from tool.geodata import *

class ToolProvider:
    """Provides a toolbox for the agents to use."""

    def __init__(self, municipality_name: str) -> None:
        """
        Initialize the ToolProvider.

        Args:
            municipality_name (str): The name of the municipality for which tools are provided.
        """
        tool_registry: dict[str, dict[str, StructuredTool]] = {
            "potential": _potential_tools(municipality_name)
        }
        self._build_vector_store(tool_registry)
        # store flattened tool registry
        # indexed by ID to allow further
        # retrieval of the tool
        self._tool_registry: dict[str, StructuredTool] = {
            tool_id: tool
            for category_tools in tool_registry.values()
            for tool_id, tool in category_tools.items()
        }
        self._tool_registry_by_name: dict[str, StructuredTool] = {
            tool.name: tool
            for category_tools in tool_registry.values()
            for _, tool in category_tools.items()
        }

    def _build_vector_store(self, tool_registry: dict[str, dict[str, StructuredTool]]) -> None:
        # instantiate and populate
        # vector store from tools
        self._vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model="nomic-embed-text:v1.5"))
        # doing this at runtime may
        # be beneficial for runtime
        # tools adding support
        documents = [
            Document(
                page_content=tool.description,
                id=id,
                metadata={
                    "category": category,
                    "tool_name": tool.name
                }
            )
            for category, tools in tool_registry.items()
            for id, tool in tools.items()
        ]
        self._vector_store.add_documents(documents)

    def get(self, name: str) -> Optional[StructuredTool]:
        """
        Get a StructuredTool from its associated name.

        Args:
            name (str): The tool's associated name.

        Returns:
            Optional[StructuredTool]: The tool, if it is indexed in the tool registry.
        """
        return self._tool_registry_by_name.get(name, None)

    def search(self, query: str, filter: Optional[Callable[[Document], bool]] = None) -> list[StructuredTool]:
        """
        Search for a StructuredTool matching the query and filter.

        Args:
            query (str): The search query string.
            filter (Callable[[Document], bool]): A callable that takes a Document and returns True if it matches the filter criteria. By default, assigned to None.

        Returns:
            list[StructuredTool]: A list of releveant StructuredTool matching the query and filter. No filtering by default.
        """
        return [
            self._tool_registry[doc.id]
            for doc in self._vector_store.similarity_search(query=query, filter=filter)
            if doc.id is not None
        ]

def _potential_tools(municipality_name: str) -> dict[str, StructuredTool]:
    """
    Returns a list of tools related to assessing the energy potential in a municipality.

    Args:
        municipality_name (str): The name of the municipality.

    Returns:
        dict[str, StructuredTool]: A dictionnary of StructuredTool objects relevant to energy potential.
    """

    return {
        str(uuid.uuid4()): tool.factory(municipality_name=municipality_name)
        for tool in [
            RoofingSolarPotentialEstimatorTool,
            RoofingSolarPotentialAggregatorTool,
            FacadesSolarPotentialEstimatorTool,
            FacadesSolarPotentialAggregatorTool,
            SmallHydroPotentialTool,
            LargeHydroPotentialTool,
            BiomassAvailabilityTool,
            HydropowerInfrastructureTool,
            WindTurbinesInfrastructureTool,
            BiogasInfrastructureTool,
            IncinerationInfrastructureTool,
            EffectiveInfrastructureTool,
            ThermalNetworksInfrastructureTool,
            SewageTreatmentPotentialTool
        ]
    }
