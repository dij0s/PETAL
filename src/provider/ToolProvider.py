import uuid

from typing import Callable, Optional
from functools import reduce
import numpy as np

from langchain_core.tools.structured import StructuredTool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_ollama import OllamaEmbeddings

from sentence_transformers import CrossEncoder
from torch.nn import Sigmoid

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
            "needs": _tools_needs(municipality_name),
            "potential": _tools_potential(municipality_name),
            "infrastructure": _tools_infrastructure(municipality_name),
        }
        await self._build_vector_store(tool_registry)
        # store flattened tool registry
        # for various lookups
        self._tools_by_category: dict[str, list[StructuredTool]] = {
            category: list(tools.values())
            for category, tools in tool_registry.items()
        }
        self._tool_registry: dict[str, StructuredTool] = {
            tool_id: tool
            for tools in tool_registry.values()
            for tool_id, tool in tools.items()
        }
        self._tool_registry_by_name: dict[str, StructuredTool] = {}
        self._category_by_tool_name: dict[str, str] = {}
        for category, tools in tool_registry.items():
            for tool in tools.values():
                self._tool_registry_by_name[tool.name] = tool
                self._category_by_tool_name[tool.name] = category
        # initialize reranking model
        self._reranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

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

    def get_category(self, name: str) -> Optional[str]:
        """
        Get the category associated to a tool.

        Args:
            name (str): The tool's associated name.

        Returns:
            Optional[str]: The category of the tool, if associated.
        """
        return self._category_by_tool_name.get(name, None)

    def get_tools(self, category_name: str) -> list[StructuredTool | Any]:
        """
        Get the tools associated to a category.

        Args:
            category_name (str): The name of the category.

        Returns:
            list[StructuredTool | Any]: The list of tools associated to that category.
        """
        return self._tools_by_category.get(category_name, [])

    def get_last_retrieved_categories(self) -> Optional[list[str]]:
        """
        Get the last retrieved categories associated to tools.

        Args:
            None

        Returns:
            Optional[list[str]]: A list of the last retrieved categories or None.
        """
        return self._last_retrieved_categories if len(self._last_retrieved_categories) > 0 else None

    async def asearch(self, query: str, max_n: int, k: int = 4, filter: Optional[Callable[[Document], bool]] = None) -> list[StructuredTool]:
        """
        Search for a StructuredTool matching the query and filter.
        First retrieves documents based on cosine similarity indicator and the applies crossencoder reranking.

        Args:
            query (str): The search query string.
            max_n (int): The number of tools to select after crossencoder reranking.
            k (int): The number of tools to retrieve from the vector store for futher reranking.
            filter (Callable[[Document], bool]): A callable that takes a Document and returns True if it matches the filter criteria. By default, assigned to None.

        Returns:
            list[StructuredTool]: A list of releveant StructuredTool matching the query and filter. No filtering by default.
        """
        # handle invalid number of
        # documents to retrieve after
        # crossencoder reranking
        if k <= 0:
            k = 4
        max_n = max(1, max_n if max_n <= k else k)
        # retrieve batch of documents
        # using cosine similarity
        docs = await self._vector_store.asimilarity_search(query=query, k=k, filter=filter)
        # apply crossencoder reranking
        # and use sigmoid activation
        # function for probability (single class)
        # finally apply softmax for
        # easier thresholding
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._reranking_model.predict(pairs, activation_fn=Sigmoid())
        scores = scores * (1 / sum(scores))
        # retrieve best documents
        threshold = 1 / len(scores)
        above_threshold_indices = [index for index, score in enumerate(scores) if score > threshold]
        top_indices = sorted(above_threshold_indices, key=lambda index: scores[index], reverse=True)[:min(len(above_threshold_indices), max_n)]
        top_docs = [docs[i] for i in top_indices]
        # get top tools and store
        # their categories for
        # future lookup when guiding
        # the user
        top_tools = [
            self._tool_registry[doc.id]
            for doc in top_docs
        ]
        self._last_retrieved_categories = list(set([
            self._category_by_tool_name[tool.name]
            for tool in top_tools
        ]))

        return top_tools

def _tools_needs(municipality_name: str) -> dict[str, StructuredTool]:
    """
    Returns a list of tools related to assessing the energy needs in a municipality.

    Args:
        municipality_name (str): The name of the municipality.

    Returns:
        dict[str, StructuredTool]: A dictionnary of StructuredTool objects relevant to energy needs.
    """

    return {
        str(uuid.uuid4()): tool.factory(municipality_name=municipality_name)
        for tool in [
            HeatingCoolingNeedsIndustryTool,
            HeatingCoolingNeedsHouseholdsTool,
            EnergyNeedsTool,
            BuildingsConstructionPeriodsTool,
            BuildingsEmissionEnergySourcesTool
        ]
    }

def _tools_potential(municipality_name: str) -> dict[str, StructuredTool]:
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
            FacadesSolarPotentialEstimatorTool,
            SmallHydroPotentialTool,
            LargeHydroPotentialTool,
            BiomassAvailabilityTool,
            WastewaterTreatmentPotentialTool,
        ]
    }

def _tools_infrastructure(municipality_name: str) -> dict[str, StructuredTool]:
    """
    Returns a list of tools related to assessing the energy infrastructure in a municipality.

    Args:
        municipality_name (str): The name of the municipality.

    Returns:
        dict[str, StructuredTool]: A dictionnary of StructuredTool objects relevant to energy infrastructure.
    """

    return {
        str(uuid.uuid4()): tool.factory(municipality_name=municipality_name)
        for tool in [
            HydropowerInfrastructureTool,
            WindTurbinesInfrastructureTool,
            BiogasInfrastructureTool,
            IncinerationInfrastructureTool,
            EffectiveInfrastructureTool,
            ThermalNetworksInfrastructureTool,
        ]
    }
