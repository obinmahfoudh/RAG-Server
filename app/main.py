import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ingestion_routes import router as ingest_router
from app.api.query_routes import router as query_router
import uvicorn

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

app.include_router(ingest_router, prefix="/api")
app.include_router(query_router, prefix="/api")

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "rag-backend"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:main:app", host="0.0.0.0", port=port, reload=False)