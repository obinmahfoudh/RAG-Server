import sys
import os
import asyncio
from app.services.ingestion import IngestionService, get_embeddings
from langchain_chroma import Chroma
from reportlab.pdfgen import canvas 
import pickle

def create_dummy_pdf(filename: str):
    """Creates a temporary 2-page PDF for testing."""
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "The only thing that I know")
    c.drawString(100, 730, "is that I know nothing.")
    c.showPage()
    
    c.drawString(100, 750, "The quick brown fox")
    c.drawString(100, 730, "jumps over the lazy dog")
    c.save()
    print(f"[+] Created dummy PDF: {filename}")

async def test_pipeline(test_pdf_path = None):
    # Check if file is provided otherwise make a simple dummy pdf file
    print("--- Testing Document Ingestion ---")
    if test_pdf_path is None:
        test_pdf_path = "test_docs.pdf"
        create_dummy_pdf(test_pdf_path)
    
    service = IngestionService()
    
    # Run the ingestion method directly 
    result = service.ingest_file(test_pdf_path, test_pdf_path)
    print(f"--- Ingestion Result: {result} --- ")
    
    print("\n--- Verifying ChromaDB (Dense Storage) ---")
    # Connect directly to the Chroma database to verify it saved
    vector_store = Chroma(
        collection_name="rag_documents",
        embedding_function= get_embeddings(),
        persist_directory=service.persist_dir
    )
    
    db_contents = vector_store.get() 
    total_vectors = len(db_contents['documents'])
    print(f"Vectors stored in ChromaDB: {total_vectors}")
    
    if total_vectors > 0:
        print(f"Sample Chunk ID: {db_contents['ids'][0]}")
        print(f"Sample Metadata: {db_contents['metadatas'][0]}")
        print(f"Sample Text: {db_contents['documents'][0][:50]}...")
    
    print("\n---Verifying BM25 Pickle File (Sparse Storage) ---")
    if os.path.exists(service.bm25_path):
        with open(service.bm25_path, "rb") as f:
            bm25_corpus = pickle.load(f)
        print(f"BM25 chunks stored: {len(bm25_corpus)}")
        if len(bm25_corpus) > 0:
            print(f"Sample Metadata: {bm25_corpus[0]['metadata']}")
            print(f"Sample BM25 Content: {bm25_corpus[0]['page_content'][:50]}")
    else:
        print("BM25 pickle file not found!")
        
    # Delete test pdf file. Can comment out to keep
    if os.path.exists(test_pdf_path) and test_pdf_path == "test_docs.pdf":
        os.remove(test_pdf_path)

if __name__ == "__main__":
    try:
        args = sys.argv[1]
        asyncio.run(test_pipeline(args))
    except IndexError:
        asyncio.run(test_pipeline())