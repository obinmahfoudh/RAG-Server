from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.services.retriever import HybridRetriever
from app.services.llm import LLMService

router = APIRouter(tags=["Query"])
retriever = HybridRetriever()
llm_service = LLMService()

# Formats request
# Provides example for query string
# Sets limits for top_k int
class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What does the quick brown fox do?"])
    top_k: int = Field(default=4, ge=1, le=10)

@router.post("/query")
async def query_documents(payload: QueryRequest):
    """
    Executes a hybrid search across dense (ChromaDB) and sparse (BM25) indexes,
    then streams the grounded LLM response using Server-Sent Events (SSE).
    """
    retrieved_docs = retriever.hybrid_search(
        query=payload.query,
        top_k=payload.top_k,
        candidate_pool=10
    )

    return StreamingResponse(
        llm_service.stream_answer(query=payload.query, docs=retrieved_docs),
        media_type="text/event-stream"
    )