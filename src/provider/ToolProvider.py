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

    _instances: dict[str, "ToolProvider"] = {}
    _locks: dict[str, asyncio.Lock] = {}

    _last_retrieved_tools: list[StructuredTool] = []

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use ToolProvider.acreate(municipality_name) to instantiate.")

    @classmethod
    async def acreate(cls, municipality_name: str) -> "ToolProvider":
        # singleton logic
        if municipality_name not in cls._instances:
            if municipality_name not in cls._locks:
                cls._locks[municipality_name] = asyncio.Lock()
            async with cls._locks[municipality_name]:
                if municipality_name not in cls._instances:
                    # create a new instance
                    # if no such one
                    self = object.__new__(cls)
                    await self._ainit(municipality_name)
                    cls._instances[municipality_name] = self
        return cls._instances[municipality_name]

    async def _ainit(self, municipality_name: str):
        tool_registry: dict[str, dict[str, StructuredTool]] = {
            "potential": _potential_tools(municipality_name)
        }
        await self._build_vector_store(tool_registry)
        # store flattened tool registry
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

    async def _build_vector_store(self, tool_registry: dict[str, dict[str, StructuredTool]]) -> None:
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
        await self._vector_store.aadd_documents(documents)

    def get(self, name: str) -> Optional[StructuredTool]:
        """
        Get a StructuredTool from its associated name.

        Args:
            name (str): The tool's associated name.

        Returns:
            Optional[StructuredTool]: The tool, if it is indexed in the tool registry.
        """
        return self._tool_registry_by_name.get(name, None)

    def get_last_retrieved_tools(self) -> Optional[list[StructuredTool]]:
        """
        Get the last retrieve tools.

        Args:
            None

        Returns:
            Optional[list[StructuredTools]]: A list of the last retrieved tools or None.
        """
        return self._last_retrieved_tools if len(self._last_retrieved_tools) > 0 else None

    def remove_used_tool(self, tool: StructuredTool) -> None:
        """
        Removes a used tool from the last retrieved tools. Allows to not recommend it anymore.

        Args:
            name (StructuredTool): The tool to remove.

        Returns:
            None
        """
        if tool in self._last_retrieved_tools:
            self._last_retrieved_tools.remove(tool)
        else:
            return

    async def asearch(self, query: str, k: int = 4, filter: Optional[Callable[[Document], bool]] = None) -> list[StructuredTool]:
        """
        Search for a StructuredTool matching the query and filter.

        Args:
            query (str): The search query string.
            k (int): The number of tools to retrieve.
            filter (Callable[[Document], bool]): A callable that takes a Document and returns True if it matches the filter criteria. By default, assigned to None.

        Returns:
            list[StructuredTool]: A list of releveant StructuredTool matching the query and filter. No filtering by default.
        """
        docs = await self._vector_store.asimilarity_search(query=query, k=k, filter=filter)
        # store last retrieved tools to
        # easily direct the user into
        # similar data to what is requested
        self._last_retrieved_tools = [
            self._tool_registry[doc.id]
            for doc in docs
            if doc.id is not None
        ]
        return self._last_retrieved_tools

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
            SewageTreatmentPotentialTool,
            BuildingsConstructionPeriodsTool,
            HeatingCoolingNeedsIndustryTool,
            HeatingCoolingNeedsHouseholdsTool,
            BuildingsEmissionEnergySourcesTool,
            EnergyNeedsTool
        ]
    }
