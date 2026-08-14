import os
from fastapi import APIRouter, HTTPException
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from services.langchain_rag_services.rag_ingestion_service import rag_ingestion_service

load_dotenv()

TABLE_NAME = "aws_rag_documents"
DATABASE_URL = os.getenv("DATABASE_URL")

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
# users what topics are covered. Metadata is stored as JSONB in the cmetadata column.
@router.get("/documents")
def get_ingested_documents():
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT DISTINCT (cmetadata->>'source') as source
                    FROM {TABLE_NAME}
                    WHERE cmetadata->>'source' IS NOT NULL
                """)
                rows = cur.fetchall()
                filenames = {Path(row[0]).name for row in rows}
                return {
                    "documents": sorted(filenames)
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Wipes the table so you can re-ingest cleanly (e.g. after fixing a chunking bug)
# without duplicate-ID errors from old records. If the table doesn't exist, that's
# fine -- we just return successfully.
@router.delete("/")
def delete_all():
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
                conn.commit()
        return {"message": f"Table '{TABLE_NAME}' cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
