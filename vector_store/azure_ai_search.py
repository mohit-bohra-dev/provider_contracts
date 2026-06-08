"""Azure AI Search vector store adapter."""
from __future__ import annotations

from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from ._base import AbstractVectorStoreProvider, SearchResult, VectorDocument


_DEFAULT_ENDPOINT = "https://YOUR-SEARCH-SERVICE.search.windows.net"
_DEFAULT_INDEX = "default"


class AzureAISearchVectorStoreProvider(AbstractVectorStoreProvider):
    """Wraps Azure AI Search for vector storage and retrieval.

    Requires the ``azure-search-documents`` optional extra::

        pip install provider-contracts[azure-search]
    """

    def __init__(
        self,
        endpoint: str = _DEFAULT_ENDPOINT,
        api_key: str | None = None,
        index_name: str = _DEFAULT_INDEX,
        embedding_dimensions: int = 1536,
    ) -> None:
        """Initialize the Azure AI Search vector store provider.

        Args:
            endpoint: Azure AI Search service endpoint
            api_key: Azure AI Search admin API key
            index_name: Name of the search index to use
            embedding_dimensions: Dimensions of the embedding vectors
        """
        if not api_key:
            raise ValueError("api_key is required for Azure AI Search provider")
            
        self._endpoint = endpoint
        self._api_key = api_key
        self._index_name = index_name
        self._embedding_dimensions = embedding_dimensions
        
        # Create clients
        credential = AzureKeyCredential(api_key)
        self._search_client = SearchClient(
            endpoint=endpoint, index_name=index_name, credential=credential
        )
        self._index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        
        # Ensure index exists
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the search index if it doesn't exist."""
        try:
            # Try to get the index - if it exists, we're done
            self._index_client.get_index(self._index_name)
        except Exception:
            # Index doesn't exist, create it
            self._create_index()

    def _create_index(self) -> None:
        """Create the search index with vector support."""
        # Define fields
        fields = [
            SearchField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True,
                retrievable=True,
            ),
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._embedding_dimensions,
                vector_search_profile_name="myHnswProfile",
            ),
            SearchField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
                filterable=True,
            ),
        ]

        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ],
        )

        # Create the index
        index = SearchIndex(
            name=self._index_name,
            fields=fields,
            vector_search=vector_search,
        )
        
        self._index_client.create_index(index)

    async def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        _ = (name, distance)
        if vector_size != self._embedding_dimensions:
            raise ValueError(
                f"vector_size {vector_size} does not match configured "
                f"embedding_dimensions {self._embedding_dimensions}"
            )

    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        """Upsert documents into the vector store.
        
        Note: Namespace is ignored in this implementation as Azure AI Search
        uses a single index. For multi-tenancy, consider using metadata filtering.
        """
        # Convert VectorDocument to Azure Search document format
        search_docs = []
        for doc in documents:
            search_doc = {
                "id": doc.id,
                "content": doc.text or "",
                "embedding": doc.embedding,
                "metadata": str(doc.metadata) if doc.metadata else "{}",
            }
            search_docs.append(search_doc)
        
        # Upload documents
        self._search_client.upload_documents(documents=search_docs)

    async def search(
        self,
        embedding: list[float],
        *,
        namespace: str = "default",
        top_k: int = 5,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using vector similarity."""
        # Create vector query
        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )
        
        # Build filter string if provided
        filter_str = None
        if filter:
            filter_parts = []
            for key, value in filter.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} eq '{value}'")
                else:
                    filter_parts.append(f"{key} eq {value}")
            filter_str = " and ".join(filter_parts)
        
        # Perform search
        results = self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_str,
            select=["id", "content", "metadata"],
            top=top_k,
        )
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            score = result["@search.score"] if "@search.score" in result else 0.0
            if score >= min_score:
                # Parse metadata back to dict if possible
                metadata = {}
                if "metadata" in result and result["metadata"]:
                    try:
                        import json
                        metadata = json.loads(result["metadata"])
                    except Exception:
                        metadata = {"raw": result["metadata"]}
                
                search_results.append(
                    SearchResult(
                        id=result["id"],
                        score=float(score),
                        metadata=metadata,
                        text=result.get("content", ""),
                    )
                )
        
        return search_results

    async def delete(self, ids: list[str], *, namespace: str = "default") -> None:
        """Delete documents by ID.
        
        Note: Namespace is ignored in this implementation.
        """
        # Delete documents by ID
        self._search_client.delete_documents(documents=[{"id": doc_id} for doc_id in ids])