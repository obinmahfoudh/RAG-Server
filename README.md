This is a simple RAG application that ingests pdf files.

## Methodology

This rag application uses dense and sparse search and combines them using Reciprocal Rank Fusion to provide an llm relevant context. Dense search is done using an embedding model and ChromaDB to store the embedded chunks. Sparse search is done using
BM25 and stored using pickle files. These searches are then combined using Reciprocal Rank Fusion using the formula: `RRF_Score = SUM( 1.0 / (rrf_k + rank_i))`

where rrf_k is a constant set to 60, which seems to be best default according to studies and rank_i is the rank of chunk i based on whatever search.

These scores are then added for each relevant chunk and this final score is used to rank and retrieve top-k elements. Depending on use cases you can weigh different 
search methods to best fit your needs. For example you could give 70% weight to sparse search results and 30% weight to dense search results and add those
to get your final ranking. Since this is a simple demo just to familiarize myself with this material I just added them, so equal weights for both.


## Changes
Initially I had the application working fine locally but only through a script. I wanted to host it and provide a front-end to interact with it and so, had to make a lot of changes.

Initially I used all-MiniLM-L6-v2 for my embeddings and llama3.2 from ollama as my llm. Since I was hosting the backend on render
I was running into issues where loading the sentence-transformers package to use all-miniLM was causing out of memory issues on the free tier.
So in order to get the backend running I had to substitute it for BAAI/bge-small-en-v1.5. For ollama I substituted to groqs openai/gpt-oss-120b.

If you dont provide environment variables for the groq api key and the embedding model name it will default to the local version of using ollama as the llm and the all-MiniLM embedding model
# Demo
A running demo of the project is available at: https://rag-server.streamlit.app/


