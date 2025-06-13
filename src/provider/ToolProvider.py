import os
import asyncio
import uuid
import numpy as np

from typing import Callable, Optional, Awaitable
from functools import reduce

from langchain_core.tools.structured import StructuredTool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_ollama import OllamaEmbeddings

from langchain_redis import RedisVectorStore
from redisvl.schema import IndexSchema

from sentence_transformers import CrossEncoder

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
        EMBEDDING_MODEL = os.getenv("OLLAMA_MODEL_EMBEDDING", "nomic-embed-text:v1.5")
        INDEX_NAME = os.getenv("REDIS_INDEX_CONSTRAINTS", "idx:doc_vss")
        REDIS_URL = os.getenv("REDIS_URL_CONSTRAINTS", "redis://localhost:6379")

        # instantiate and populate
        # vector store for runtime
        # tools semantic search
        embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self._vector_store_tools = InMemoryVectorStore(embedding=embedder)
        # instantiate redis vector
        # store embedder
        self._vector_store_constraints = RedisVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embedder,
            redis_url=REDIS_URL,
            embedding_field="vector",
            content_field="chunk_content",
        )

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
        await self._vector_store_tools.aadd_documents(documents)

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

    async def asearch(
        self,
        query: str,
        max_n_tools: int,
        k_tools: int):
        """
        Search for StructuredTool objects and constraining document chunks matching the query and filter.

        Args:
            query (str): The search query string.
            max_n_tools (int): The maximum number of tools to select after crossencoder reranking.
            k_tools (int): The number of tools and constraining chunks to retrieve from the vector store for futher reranking.

        Returns:
            tuple[list[StructuredTool], list[str]]: A tuple containing a list of relevant StructuredTool objects that match the query and filter, along with the the constraining document chunks. By default, no filtering is applied.
        """
        tools_task = self._asearch_tools(query, max_n_tools, k_tools)
        constraints_task = self._asearch_constraints(query)
        # run both results
        # asynchronously and
        # then only gather
        # maybe handle query
        # embedding differently ?
        return await asyncio.gather(tools_task, constraints_task)

    async def _rerank_documents(self, query: str, docs: list[Document], max_n: int, batch_size: int = 5) -> list[Document]:
        """
        Rerank a list of documents based on their relevance to the query using the cross-encoder model.

        Args:
            query (str): The search query string.
            docs (list[Document]): The list of documents to rerank.
            max_n (int): The maximum number of top documents to return.
            batch_size (int): The batch size for processing documents during reranking. Defaults to 5.

        Returns:
            list[Document]: A list of the top reranked documents that meet the threshold, or an empty list if no documents meet the criteria.
        """
        if not docs:
            return []
        # apply crossencoder reranking
        pairs = [(query, doc.page_content) for doc in docs]
        # predict by batches for
        # faster inference
        logits = self._reranking_model.predict(pairs, batch_size=batch_size)
        # apply softmax normalization and
        # select documents from threshold
        # set in a way to extract "outliers"
        exp_logits = np.exp(logits - np.max(logits))
        scores = exp_logits / np.sum(exp_logits)
        threshold = np.mean(scores) + np.std(scores)
        selected_indices = np.where(scores > threshold)[0]
        # detect uniform distribution
        # using coefficient of variation
        cv = np.std(scores) / np.mean(scores)
        uniformity_threshold = 0.15
        if cv < uniformity_threshold:
            # take top max_n directly
            top_indices = np.argsort(scores)[::-1][:max_n]
            return [docs[index] for index in top_indices]
        else:
            # take top max_n from the
            # thresholded results
            selected_scores = scores[selected_indices]
            top_indices = np.argsort(selected_scores)[::-1][:max_n]
            return [docs[selected_indices[index]] for index in top_indices]

    async def _asearch_tools(self, query: str, max_n: int, k: int = 5, filter: Optional[Callable[[Document], bool]] = None) -> list[StructuredTool]:
        """
        Search for StructuredTool objects matching the query and filter.
        First retrieves documents based on cosine similarity indicator and the applies crossencoder reranking.

        Args:
            query (str): The search query string.
            max_n (int): The number of tools to select after crossencoder reranking.
            k (int): The number of tools to retrieve from the vector store for futher reranking. Default to 5.
            filter (Callable[[Document], bool]): A callable that takes a Document and returns True if it matches the filter criteria. By default, assigned to None.

        Returns:
            list[StructuredTool]: A list of relevant StructuredTool objects that match the query and filter. By default, no filtering
        """
        # handle invalid number of
        # documents to retrieve after
        # crossencoder reranking
        if k <= 0:
            k = 5
        max_n = max(1, max_n if max_n <= k else k)
        # retrieve documents
        # using cosine similarity
        docs = await self._vector_store_tools.asimilarity_search(query=query, k=k, filter=filter)
        # rerank documents
        top_docs = await self._rerank_documents(query=query, docs=docs, max_n=max_n)
        # get top tools and store
        # their categories for
        # future lookup when guiding
        # the user
        top_tools = [
            self._tool_registry[doc.id] # type: ignore
            for doc in top_docs
        ]
        self._last_retrieved_categories = list(set([
            self._category_by_tool_name[tool.name]
            for tool in top_tools
        ]))

        return top_tools

    async def _asearch_constraints(self, query: str) -> list[tuple[str, str]]:
        """
        Search for constraining document chunks matching the query.

        Args:
            query (str): The search query string.

        Returns:
            list[tuple[str, str]]: A list of document contents and their corresponding sources that are relevant to the query.
        """
        # retrieve documents
        # using cosine similarity
        # manually retrieve redis client
        # and query documents by retrieved
        # identifiers asynchronously
        client = self._vector_store_constraints.config.redis()
        if not client:
            return []

        pipe = client.pipeline()
        pipe.json().mget([
            doc.metadata.get("id", "")
            for doc in await self._vector_store_constraints.asimilarity_search(query=query, k=25)
        ], "$['document_title', 'page_number', 'chunks']")
        # flatten documents into
        # both the chunks and source
        docs = reduce(
            lambda res, ds: [
                *res, *[
                    Document(
                        metadata={ "source":  d.get("source")},
                        page_content=d.get("page_content") # type: ignore
                    )
                    for d in ds
                ]
            ],
            map(lambda result: [
                {
                    "source": f"{result[0]}, page n° {result[1]}",
                    "page_content": chunk
                }
                for chunk in result[2]
            ], (await asyncio.get_event_loop().run_in_executor(None, pipe.execute))[0]),
            []
        )
        # rerank chunks from
        # retrieved documents
        top_docs = await self._rerank_documents(query=query, docs=docs, max_n=10)
        return [
            (doc.page_content, doc.metadata.get("source", ""))
            for doc in top_docs
        ]


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
            HeatingCoolingNeedsHouseholdsServicesTool,
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
            EffectiveRenewableEnergiesTool,
            ThermalNetworksInfrastructureTool,
        ]
    }
