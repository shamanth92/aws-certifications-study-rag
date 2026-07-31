from fastapi import APIRouter, HTTPException
from pathlib import Path
import chromadb
from services.langchain_rag_services.rag_ingestion_service import rag_ingestion_service

# Separate chromadb client here (distinct from services/rag_services/vector_store_service.py's
# client) purely for the delete endpoint below -- LangChain's Chroma wrapper
# doesn't expose collection deletion, so we talk to the underlying chromadb
# client directly, targeting the same collection name the ingestion service uses.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "aws-rag-documents"

router = APIRouter(prefix="/langchain/ingestion", tags=["langchain-ingestion"])


# Runs the LangChain ingestion pipeline over every PDF in docs_dir, one file
# at a time (each call embeds+stores that file's chunks immediately).
@router.post("/")
def ingest(docs_dir: str = "docs"):
    if not Path(docs_dir).exists():
        raise HTTPException(status_code=404, detail=f"Directory '{docs_dir}' not found")

    pdf_files = list(Path(docs_dir).glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(status_code=404, detail=f"No PDF files found in '{docs_dir}'")

    try:
        total_chunks = 0
        for pdf_path in pdf_files:
            result = rag_ingestion_service(str(pdf_path))
            total_chunks += result["chunks_created"]

        return {
            "documents_processed": len(pdf_files),
            "chunks_created": total_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Returns the distinct filenames currently ingested, for the frontend to show
# users what topics are covered. LangChain's loader stores the file path under
# the "source" metadata key (not "document_name") -- Path(...).name strips it
# down to just the filename.
@router.get("/documents")
def get_ingested_documents():
    try:
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        result = collection.get(include=["metadatas"])
        filenames = {Path(m["source"]).name for m in result["metadatas"]}
        return {
            "documents": sorted(filenames)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Wipes and recreates the collection so you can re-ingest cleanly (e.g. after
# fixing a chunking bug) without duplicate-ID errors from old records.
# delete_collection() raises if the collection doesn't exist yet (e.g. first
# run before any ingestion) -- that's expected and safely ignored here.
@router.delete("/")
def delete_all():
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    chroma_client.get_or_create_collection(COLLECTION_NAME)
    return {"message": f"Collection '{COLLECTION_NAME}' cleared"}
