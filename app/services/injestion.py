import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
BM25_STORE_PATH = os.getenv("BM25_STORE_PATH", "./data/bm25_corpus.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy-loaded embedding model instance
_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

# Ingestion Service Class
class IngestionService:
    def __init__(self):
        self.persist_dir = CHROMA_PERSIST_DIR
        self.bm25_path = BM25_STORE_PATH
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.bm25_path)).mkdir(parents=True, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    # 
    def load_pdf(self, file_path: str) -> List[Document]:
        """Use PyPDFLoader to open pdf file"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    # Creates unique hash for chunked content
    # Tradeoff for using source name. On one hand if we upload a version 2 of the same document then repeated chunks will still be saves 
    # but if we do not use source name then if we have multiple sources with the same content and we want to show all the unique sources with the same content we can't
    def generate_chunk_id(self, content: str, source: str, page: int, chunk_idx: int) -> str:
        """Generates a SHA-256 hash ID to prevent duplicate chunks."""
        unique_str = f"{source}:{page}:{chunk_idx}:{content}"
        return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
    
    def process_and_chunk(self, raw_docs: List[Document], filename: str) -> List[Document]:
        """Splits raw pages into structured chunks and ids them."""
        processed_chunks: List[Document] = []
        # Enumerate through pages in docs
        for page_idx, doc in enumerate(raw_docs):
            # Add 1 so index starts at 1
            page_number = doc.metadata.get("page", page_idx) + 1  
            page_splits = self.text_splitter.split_text(doc.page_content)
            # Enumerate through page chunks and clean text if there is any
            for split_idx, chunk_text in enumerate(page_splits):
                cleaned_text = chunk_text.strip()
                if not cleaned_text:
                    continue
                # ID chunk
                chunk_id = self.generate_chunk_id(cleaned_text, filename, page_number, split_idx)
                
                chunk_doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "chunk_id": chunk_id,
                        "source": filename,
                        "page": page_number,
                        "chunk_index": split_idx
                    }
                )
                processed_chunks.append(chunk_doc)

        return processed_chunks

    def update_bm25_corpus(self, new_chunks: List[Document]) -> None:
        """Appends new chunks to the persistent pickle store for BM25 sparse search."""
        existing_corpus: List[Dict[str, Any]] = []
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                existing_corpus = pickle.load(f)

        existing_ids = {doc["metadata"]["chunk_id"] for doc in existing_corpus}
        
        for chunk in new_chunks:
            if chunk.metadata["chunk_id"] not in existing_ids:
                existing_corpus.append({
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                })

        with open(self.bm25_path, "wb") as f:
            pickle.dump(existing_corpus, f)

    def index_documents(self, chunks: List[Document]) -> int:
        """Embeds and indexes chunks into ChromaDB."""
        if not chunks:
            return 0

        embeddings = get_embeddings()
        vector_store = Chroma(
            collection_name="rag_documents",
            embedding_function=embeddings,
            persist_directory=self.persist_dir
        )

        chunk_ids = [doc.metadata["chunk_id"] for doc in chunks]
        
        # Ingest into dense vector store
        vector_store.add_documents(documents=chunks, ids=chunk_ids)

        # Update sparse corpus
        self.update_bm25_corpus(chunks)

        return len(chunks)

    def ingest_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Starts end-to-end extraction, chunking, and indexing on sparse and dense storage."""
        raw_docs = self.load_pdf(file_path)
        chunks = self.process_and_chunk(raw_docs, filename)
        indexed_count = self.index_documents(chunks)

        return {
            "filename": filename,
            "total_pages": len(raw_docs),
            "chunks_created": indexed_count,
            "status": "completed"
        }