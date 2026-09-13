# '''
# استفاده از FAISS (پیشنهادی)
#
# '''
# from core.embeddings import EmbeddingGenerator
# from core.vector_store import FAISSVectorStore
#
# # ۱. ساخت generator و store
# generator = EmbeddingGenerator()
# store = FAISSVectorStore(embedding_dim=generator.embedding_dimension)
#
# # ۲. افزودن embeddingها
# embeddings = generator.embed_chunks(chunks)
# store.add_embeddings(chunks, embeddings)
#
# # ۳. ذخیره
# store.save("data/processed/vector_store")
#
# # ۴. بارگذاری (در session بعدی)
# store = FAISSVectorStore(embedding_dim=768)
# store.load("data/processed/vector_store")
#
# # ۵. جستجو
# query = "قرارداد چند ساله است؟"
# query_embedding = generator.embed_text(query)
# results = store.search(query_embedding, top_k=3)
#
# for chunk, score in results:
#     print(f"Score: {score:.3f} | {chunk.text[:100]}...")
#
# '''
# استفاده از ChromaDB (اختیاری)
#
# '''
#
# from core.embeddings import EmbeddingGenerator
# from core.vector_store import ChromaVectorStore
#
# # ۱. ساخت generator و store
# generator = EmbeddingGenerator()
# store = ChromaVectorStore(
#     collection_name="my_documents",
#     persist_directory="data/processed/chroma"
# )
#
# # ۲. افزودن embeddingها
# embeddings = generator.embed_chunks(chunks)
# store.add_embeddings(chunks, embeddings)
#
# # ۳. جستجو (persist خودکار است)
# query = "قرارداد چند ساله است؟"
# query_embedding = generator.embed_text(query)
# results = store.search(query_embedding, top_k=3)
#
# for chunk, score in results:
#     print(f"Score: {score:.3f} | {chunk.text[:100]}...")
#
#
# '''
# استفاده از Factory (انعطاف‌پذیر)
#
# '''
# from core.vector_store import create_vector_store
# from app.config import settings
#
# # خواندن نوع store از config
# store = create_vector_store(
#     store_type=settings.vector_store_type,  # "faiss" or "chroma"
#     embedding_dim=768
# )
#
# # بقیه کد یکسان است
# store.add_embeddings(chunks, embeddings)
# results = store.search(query_embedding, top_k=3)

# tests/test_vector_store.py

import pytest
import numpy as np
from core.vector_store import FAISSVectorStore


@pytest.fixture
def store():
    """Vector store."""
    return FAISSVectorStore(embedding_dim=768)


def test_add_and_search(store, sample_chunks):
    """تست افزودن و جستجو."""
    embeddings = [np.random.rand(768) for _ in sample_chunks]

    store.add_embeddings(sample_chunks, embeddings)

    query_embedding = np.random.rand(768)
    results = store.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert all(0 <= idx < len(sample_chunks) for idx, _ in results)


def test_save_load(store, sample_chunks, temp_dir):
    """تست ذخیره و بارگذاری."""
    embeddings = [np.random.rand(768) for _ in sample_chunks]
    store.add_embeddings(sample_chunks, embeddings)

    save_path = temp_dir / "test_store"
    store.save(str(save_path))

    new_store = FAISSVectorStore(embedding_dim=768)
    new_store.load(str(save_path))

    assert len(new_store.chunks) == len(sample_chunks)
