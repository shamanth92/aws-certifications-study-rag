from fastapi import FastAPI
from pydantic import BaseModel
from routers.basicRag.ingestion import router as ingestion_router
from routers.basicRag.rag_chat import router as rag_chat_router
from routers.langChainRag.langchain_ingestion import router as langchain_ingestion_router
from routers.langChainRag.langchain_rag_chat import router as langchain_rag_chat_router
from fastapi.middleware.cors import CORSMiddleware

# This app implements the same RAG (Retrieval-Augmented Generation) pipeline twice,
# side by side, for learning purposes:
#   1. "basicRag"    -> hand-rolled pipeline (no LangChain), see services/rag_services/
#   2. "langChainRag" -> same pipeline built with LangChain, see services/langchain_rag_services/
# Both ingest PDFs into their own ChromaDB collection and answer questions using
# retrieved context + an OpenAI chat model. Compare the two implementations to see
# what LangChain abstracts away (prompt chaining, retriever interface, etc.).

# uv run fastapi dev main.py

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id"],
)

# Each router is self-contained: it owns its own prefix, tags, and error handling.
# See the router files themselves for the actual endpoints.
app.include_router(ingestion_router)
app.include_router(rag_chat_router)
app.include_router(langchain_ingestion_router)
app.include_router(langchain_rag_chat_router)
