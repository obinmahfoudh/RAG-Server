import json
import os
from typing import AsyncGenerator, List
from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# For online hosting
MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MODEL_API_KEY = os.getenv("GROQ_API_KEY")
# For local 
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") 

SYSTEM_PROMPT = """You are an expert technical assistant answering questions based strictly on the provided context.

Context:
{context}

Question:
{question}

Instructions:
1. Answer the user's question using ONLY the provided context above.
2. For every factual claim, reference the source document and page number inline (e.g., [source: filename.pdf, p. 3]).
3. If the context does not contain sufficient information to answer the question, clearly state: "I cannot find relevant context in the provided documents to answer this question."
4. Do not assume or extrapolate beyond what is explicitly written in the context.

Answer:"""

class LLMService:
    def __init__(self):
        # Use model key for online hosting and ollama for local hosting
        if MODEL_API_KEY:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model= MODEL_NAME,
                api_key= SecretStr(MODEL_API_KEY),
                temperature= 0.1
            )
        else:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model= OLLAMA_MODEL,
                base_url= OLLAMA_BASE_URL,
                temperature= 0.1
            )

        self.prompt_template = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )
    # Using markdown delimiters. Could use xml as that seems to be standard for antrhoppic but I'll look into it more.
    def format_context(self, docs: List[Document]) -> str:
        formatted_blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            formatted_blocks.append(
                f"--- [Document Chunk {i} | Source: {source} | Page: {page}] ---\n{doc.page_content}"
            )
        return "\n\n".join(formatted_blocks)

    def extract_citations(self, docs: List[Document]) -> List[dict]:
        """Get citations from docs we got from rag search"""
        citations = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                citations.append({"source": source, "page": page})
        return citations

    async def stream_answer(self, query: str, docs: List[Document]) -> AsyncGenerator[str, None]:
        """
        Streams Server-Sent Events (SSE) formatted tokens and appends metadata citations at the end.
        """
        # If we get no relevant docs from rag search then return nothing
        if not docs:
            yield f"data: {json.dumps({'token': 'No relevant documents found in the database.'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        # Format the context we got from rag search then give it to the query
        formatted_context = self.format_context(docs)
        prompt = self.prompt_template.format(context=formatted_context, question=query)

        # Stream tokens from LLM
        async for chunk in self.llm.astream(prompt):
            token_data = {"token": chunk.content}
            yield f"data: {json.dumps(token_data)}\n\n"

        # Append citations
        citations = self.extract_citations(docs)
        citations_data = {"citations": citations}
        yield f"data: {json.dumps(citations_data)}\n\n"
        yield "data: [DONE]\n\n"