from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ingestion_routes import router as ingest_router
from app.api.query_routes import router as query_router

app = FastAPI(
    title="Enterprise RAG Search Engine",
    version="1.0.0",
    description="asynchronous RAG engine with hybrid search and streaming citations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "rag-backend"}