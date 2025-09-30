import os
import re
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff

class VectorStore:
    """
    Create vector store, ingest documents, retrieve relevant documents
    """
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type
        self.vectorstore_local_path = config.rag.vector_local_path
        self.docstore_local_path = config.rag.doc_local_path
        self.qdrant_url = config.rag.url
        self.qdrant_api_key = config.rag.api_key

        # Use the singleton client instead of creating a new one
        # self.client = QdrantClientManager.get_client(config)
        # Debug: Log the Qdrant configuration
        self.logger.info(f"Qdrant URL: {self.qdrant_url}")
        self.logger.info(f"Qdrant API Key: {'***' + self.qdrant_api_key[-4:] if self.qdrant_api_key else 'None'}")

        # Choose between local and cloud Qdrant
        if self.qdrant_url and self.qdrant_api_key:
            # Use Qdrant Cloud
            self.logger.info(f"Using Qdrant Cloud at: {self.qdrant_url}")
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60
            )
            self.use_local = False
        else:
            # Use Local Qdrant
            self.logger.info(f"Using Local Qdrant at: {self.vectorstore_local_path}")
            self.client = QdrantClient(path=self.vectorstore_local_path)
            self.use_local = True

    def _does_collection_exist(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        collection_info = self.client.get_collections()
        collection_names = [collection.name for collection in collection_info.collections]
        return self.collection_name in collection_names

    def _create_collection(self):
        """Create a new collection with dense and sparse vectors for hybrid retrieval."""
        # Delete existing collection if it exists
        if self._does_collection_exist():
            self.logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Successfully deleted old collection")

        # Create new collection with hybrid support
        self.logger.info(f"Creating new collection with hybrid support")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        self.logger.info(f"Created new collection with hybrid support: {self.collection_name}")

        # Verify sparse vectors are configured
        verify_info = self.client.get_collection(self.collection_name)
        sparse_check = verify_info.config.params.sparse_vectors
        self.logger.info(f"Verified sparse vectors: {list(sparse_check.keys())}")
            
    def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        """
        Load existing vectorstore and docstore for retrieval operations without ingesting new documents.

        Returns:
            Tuple containing (vectorstore, docstore)
        """
        # Check if collection exists
        print(f"🔍 Checking collection: {self.collection_name}")
        self.logger.info(f"Loading collection: {self.collection_name}")

        # Verify collection has proper sparse vector configuration
        collection_info = self.client.get_collection(self.collection_name)
        print(f"   📊 Collection points: {collection_info.points_count}")
        sparse_config = collection_info.config.params.sparse_vectors
        print(f"   🔧 Sparse vectors: {list(sparse_config.keys())}")
        self.logger.info(f"Collection has sparse vectors: {list(sparse_config.keys())}")

        # Setup sparse embeddings for hybrid retrieval
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Initialize vector store with hybrid retrieval
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="",  # Empty string for unnamed dense vector
            sparse_vector_name="sparse",
        )
        self.logger.info("Successfully initialized HYBRID retrieval")

        # Document storage - use local file store for both modes for compatibility
        docstore = LocalFileStore(self.docstore_local_path)

        self.logger.info(f"Successfully loaded existing vectorstore and docstore with HYBRID mode")
        return qdrant_vectorstore, docstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, LocalFileStore, List[str]]:
        """
        Create a vector store from document chunks or upsert documents to existing store.
        
        Args:
            document_chunks: List of document chunks
            document_path: Path to the original document
            
        Returns:
            Tuple containing (vectorstore, docstore, doc_ids)
        """
        
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents
        langchain_documents = []
        for id_idx, chunk in enumerate(document_chunks):
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[id_idx],
                        # "source_path": Path(os.path.abspath(document_path)).as_uri()
                        "source_path": os.path.join("http://localhost:8000/", document_path)
                    }
                )
            )
        
        # Setup sparse embeddings for hybrid retrieval
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Check if collection exists and has data
        collection_exists = self._does_collection_exist()

        if not collection_exists:
            self._create_collection()
            self.logger.info(f"Created new collection: {self.collection_name}")
            has_data = False
        else:
            # Verify existing collection has proper sparse configuration
            collection_info = self.client.get_collection(self.collection_name)
            sparse_config = collection_info.config.params.sparse_vectors
            if not sparse_config or "sparse" not in sparse_config:
                self.logger.warning(f"Existing collection lacks sparse vectors, recreating")
                self._create_collection()  # This will delete and recreate
                has_data = False
            else:
                has_data = collection_info.points_count > 0
                if has_data:
                    self.logger.info(f"Collection {self.collection_name} exists with {collection_info.points_count} points and proper sparse config")
                else:
                    self.logger.info(f"Collection {self.collection_name} exists with proper sparse config but empty")

        # Initialize vector store with hybrid retrieval
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="",  # Empty string for unnamed dense vector
            sparse_vector_name="sparse",
        )
        self.logger.info("Successfully initialized HYBRID retrieval for ingestion")

        # Log cloud usage
        if not self.use_local:
            if has_data:
                self.logger.info("Using cloud Qdrant for document retrieval")
            else:
                self.logger.info("Using cloud Qdrant for document ingestion")

        # Document storage for parent documents
        docstore = LocalFileStore(self.docstore_local_path)

        # Only ingest documents if collection is empty
        if not has_data:
            self.logger.info(f"Ingesting {len(langchain_documents)} documents into {self.collection_name}")
            qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)
        else:
            self.logger.info(f"Collection {self.collection_name} already has data, skipping ingestion")

        # Store document chunks in docstore as backup
        if document_chunks:  # Only store if we have chunks
            encoded_chunks = [chunk.encode('utf-8') for chunk in document_chunks]
            docstore.mset(list(zip(doc_ids, encoded_chunks)))

    def retrieve_relevant_chunks(
            self,
            query: str,
        ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on a single query.

        Args:
            query: User query

        Returns:
            List of dictionaries with content and score
        """
        return self._retrieve_single_query(query)
    
    def retrieve_with_multiple_queries(
            self,
            queries: List[str],
            top_k_final: int = 3
        ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using multiple sub-queries and aggregate results.

        Args:
            queries: List of sub-queries
            top_k_final: Final number of top chunks to return

        Returns:
            List of aggregated top chunks
        """
        print(f"🔍 Running multi-query retrieval with {len(queries)} queries")
        self.logger.info(f"Running multi-query retrieval with {len(queries)} queries")
        
        all_results = {}  # Use dict to deduplicate by chunk ID
        
        for i, query in enumerate(queries):
            print(f"   🔄 Sub-query {i+1}: '{query}'")
            self.logger.info(f"   Sub-query {i+1}: '{query}'")
            
            # Retrieve for this sub-query
            query_results = self._retrieve_single_query(query)
            print(f"  Found {len(query_results)} results for this sub-query")
            
            # Debug: Show top result for each sub-query
            if query_results:
                top_result = query_results[0]
                print(f"  Top result: Score={top_result['score']:.3f}, Full Content='{top_result.get('content', '')}'")
            
            # Add to aggregated results
            for doc in query_results:
                doc_id = doc.get('id', 'unknown')
                if doc_id not in all_results:
                    all_results[doc_id] = doc
                    all_results[doc_id]['query_matches'] = 1
                    all_results[doc_id]['max_score'] = doc['score']
                else:
                    # Update with better score and increment match count
                    all_results[doc_id]['query_matches'] += 1
                    all_results[doc_id]['max_score'] = max(all_results[doc_id]['max_score'], doc['score'])
        
        # Convert back to list and calculate combined scores
        aggregated_results = list(all_results.values())
        
        # Calculate final score: (max_score * 0.7) + (query_matches * 0.3)
        for doc in aggregated_results:
            query_match_score = min(doc['query_matches'] / len(queries), 1.0)  # Normalize to 0-1
            doc['combined_score'] = (doc['max_score'] * 0.7) + (query_match_score * 0.3)
        
        # Sort by combined score and return top K
        aggregated_results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_results = aggregated_results[:top_k_final]
        
        self.logger.info(f"   Aggregated {len(aggregated_results)} unique chunks, returning top {len(top_results)}")
        for i, doc in enumerate(top_results):
            self.logger.info(f"      Chunk {i+1}: Score={doc['combined_score']:.3f} (Max={doc['max_score']:.3f}, Matches={doc['query_matches']}/{len(queries)})")
        
        return top_results
    
    def _retrieve_single_query(
            self,
            query: str,
        ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on a single query.

        Args:
            query: User query

        Returns:
            List of dictionaries with content and score
        """
        # Use direct Qdrant search to get payload data
        # Get embeddings for the query
        query_embedding = self.embedding_model.embed_query(query)

        # Search with payload to get full document content
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=self.retrieval_top_k,
            with_payload=True
        )

        retrieved_docs = []

        for hit in search_results:
            payload = hit.payload
            score = hit.score

            if 'text' in payload:
                doc_content = payload['text']
                doc_id = payload.get('chunk_index', 'unknown')
                source = payload.get('source', 'unknown')
                source_path = payload.get('source', 'unknown')
            else:
                continue

            # Create document dict in the format expected by reranker
            doc_dict = {
                "id": str(doc_id),
                "content": doc_content,
                "score": score,  # Use the actual similarity score
                "source": source,
                "source_path": source_path,
            }
            retrieved_docs.append(doc_dict)

        return retrieved_docs