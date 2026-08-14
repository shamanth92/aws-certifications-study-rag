import os
from dotenv import load_dotenv
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv()

# LangChain equivalent of the basic pipeline's read -> chunk -> embed -> store
# steps, collapsed into one function since LangChain's building blocks each
# handle one of those steps internally:
#   PyMuPDF4LLMLoader           -> reads the PDF (like document_service.read_document)
#   RecursiveCharacterTextSplitter -> chunks text, splitting on paragraph/sentence
#                                     boundaries where possible (unlike the basic
#                                     pipeline's naive fixed-size slicing)
#   OpenAIEmbeddings + PGVectorStore -> embeds and stores in one call (add_documents
#                                  embeds internally, unlike the basic pipeline
#                                  where vectorization and storage are separate steps)
def rag_ingestion_service(file_path: str):
    loader = PyMuPDF4LLMLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    # Filter out empty chunks (e.g. image-only pages produce blank text) --
    # PGVector rejects add_documents() calls containing an empty embedding.
    chunks = [c for c in text_splitter.split_documents(documents) if c.page_content.strip()]

    # A file with no extractable text (e.g. all pages are scanned images)
    # ends up with zero chunks -- add_documents([]) would still try to embed
    # an empty list and raise the same "non-empty list ... got []" error, so
    # skip the store call entirely in that case.
    if not chunks:
        return {
            "file": file_path,
            "chunks_created": 0
        }

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")

    # Uses PGVector with table name "aws_rag_documents" to keep this
    # pipeline's vectors separate from the basic pipeline's "documents" table.
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=database_url,
        collection_name="aws_rag_documents",
        use_jsonb=True
    )

    return {
        "file": file_path,
        "chunks_created": len(chunks)
    }
