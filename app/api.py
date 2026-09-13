# app/api.py

"""
REST API برای DocMind Assistant با FastAPI.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List
import shutil

from app.config import get_settings
from core.loader import PDFLoader
from core.chunker import TextChunker
from core.embeddings import EmbeddingGenerator
from core.vector_store import create_vector_store
from core.retriever import Retriever
from core.llm import OpenAIClient, GeminiClient
from core.qa_chain import QAChain

app = FastAPI(
    title="DocMind Assistant API",
    description="RAG-based Q&A system for PDF documents",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
qa_chain = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    tokens_used: int = None


class IndexResponse(BaseModel):
    message: str
    chunks_count: int
    files_count: int


def get_qa_chain():
    """بارگذاری QA chain."""
    global qa_chain

    if qa_chain is None:
        settings = get_settings()

        if not Path(settings.vector_store_path).exists():
            raise HTTPException(status_code=404, detail="Vector store not found. Please index documents first.")

        generator = EmbeddingGenerator(model_name=settings.embedding_model)
        store = create_vector_store(
            store_type=settings.vector_store_type,
            embedding_dim=generator.embedding_dimension
        )
        store.load(settings.vector_store_path)

        retriever = Retriever(generator, store)

        if settings.llm_provider == "openai":
            llm = OpenAIClient(model=settings.llm_model)
        else:
            llm = GeminiClient(model=settings.llm_model)

        qa_chain = QAChain(retriever, llm, top_k=settings.top_k)

    return qa_chain


@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "service": "DocMind Assistant API"}


@app.post("/index", response_model=IndexResponse)
async def index_documents(files: List[UploadFile] = File(...)):
    """ایندکس کردن فایل‌های PDF."""
    settings = get_settings()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    # ذخیره فایل‌ها
    saved_files = []
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

        file_path = settings.raw_data_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(file_path)

    # بارگذاری و چانک کردن
    all_pages = []
    for file_path in saved_files:
        loader = PDFLoader(str(file_path))
        pages = loader.load()
        all_pages.extend(pages)

    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_pages(all_pages)

    # تولید embeddings
    generator = EmbeddingGenerator(model_name=settings.embedding_model)
    embeddings = generator.embed_chunks(chunks)

    # ذخیره در vector store
    store = create_vector_store(
        store_type=settings.vector_store_type,
        embedding_dim=generator.embedding_dimension
    )
    store.add_embeddings(chunks, embeddings)
    store.save(settings.vector_store_path)

    # Reset global chain
    global qa_chain
    qa_chain = None

    return IndexResponse(
        message="Documents indexed successfully",
        chunks_count=len(chunks),
        files_count=len(files)
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """پرسیدن سوال از اسناد."""
    chain = get_qa_chain()

    top_k = request.top_k if request.top_k else get_settings().top_k
    result = chain.ask(request.question, top_k=top_k)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        tokens_used=result.get("tokens_used")
    )


@app.delete("/index")
def clear_index():
    """پاک کردن ایندکس."""
    settings = get_settings()

    if Path(settings.vector_store_path).exists():
        shutil.rmtree(settings.vector_store_path)
    global qa_chain
    qa_chain = None

    return {"message": "Index cleared successfully"}


@app.get("/status")
def get_status():
    """وضعیت سیستم."""
    settings = get_settings()
    indexed = Path(settings.vector_store_path).exists()

    return {
        "indexed": indexed,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "vector_store_type": settings.vector_store_type
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
