import os
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from app.services.ingestion import get_embeddings, CHROMA_PERSIST_DIR, BM25_STORE_PATH
from langchain_core.documents import Document



class HybridRetriever:
    # k=60 seems to be most common default value
    def __init__(self, rrf_k: int = 60):
        self.persist_dir = CHROMA_PERSIST_DIR
        self.bm25_path = BM25_STORE_PATH
        # rrf constant (rank fusion)
        self.rrf_k = rrf_k  

    def get_vector_store(self) -> Chroma:
        """Loads chroma db from storage."""
        return Chroma(
            collection_name="rag_documents",
            embedding_function=get_embeddings(),
            persist_directory=self.persist_dir
        )

    def search_dense(self, query: str, top_k: int) -> List[Document]:
        """Dense semantic search via ChromaDB."""
        try:
            vector_store = self.get_vector_store()
            return vector_store.similarity_search(query, k=top_k)
        except Exception:
            return []

    def search_sparse(self, query: str, top_k: int) -> List[Document]:
        """Sparse keyword search via BM25."""
        if not os.path.exists(self.bm25_path):
            return []

        with open(self.bm25_path, "rb") as f:
            corpus: List[Dict[str, Any]] = pickle.load(f)

        if not corpus:
            return []

        # Tokenize corpus for BM25 matching
        tokenized_corpus = [doc["page_content"].lower().split() for doc in corpus]
        # This is just tf-idf but better
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize user query
        tokenized_query = query.lower().split()
        # Score context using query
        scores = bm25.get_scores(tokenized_query)

        # Get indices of top scoring documents
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if at least one token matched
                item = corpus[idx]
                results.append(
                    Document(
                        page_content=item["page_content"],
                        metadata=item["metadata"]
                    )
                )
        return results

    def hybrid_search(self, query: str, top_k: int = 4, candidate_pool: int = 10) -> List[Document]:
        """
        Executes dual search and merges candidate ranks using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score = SUM( 1.0 / (rrf_k + rank_i) )
        Converts highest ranking dense & sparse candidate scores into a rrf score which is then sorted to select the top k.
        """
        # Use candidate pool to get that many relevant documents from each search. Will narrow down to top_k later
        dense_docs = self.search_dense(query, top_k=candidate_pool)
        sparse_docs = self.search_sparse(query, top_k=candidate_pool)

        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}

        # Score Dense Candidates
        for rank, doc in enumerate(dense_docs, start=1):
            chunk_id = doc.metadata.get("chunk_id") or doc.page_content
            doc_lookup[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Score Sparse Candidates
        for rank, doc in enumerate(sparse_docs, start=1):
            chunk_id = doc.metadata.get("chunk_id") or doc.page_content
            doc_lookup[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort by fused score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        # Narrow documents to top_k after getting candidate pool amount from each search 
        return [doc_lookup[cid] for cid in sorted_ids[:top_k]]