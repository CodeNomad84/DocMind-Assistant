# # استفاده پایه
#
# from core.embeddings import EmbeddingGenerator
# from core.vector_store import FAISSVectorStore
# from core.retriever import Retriever
#
# # ۱. بارگذاری vector store
# generator = EmbeddingGenerator()
# store = FAISSVectorStore(embedding_dim=768)
# store.load("data/processed/vector_store")
#
# # ۲. ساخت retriever
# retriever = Retriever(
#     embedding_generator=generator,
#     vector_store=store,
#     score_threshold=0.3  # فیلتر نتایج ضعیف
# )
#
# # ۳. جستجو
# query = "قرارداد چند ساله است؟"
# results = retriever.retrieve(query, top_k=5)
#
# # ۴. نمایش نتایج
# for result in results:
#     print(result)
#     print("-" * 80)
#
# # ۵. فرمت context برای LLM
# context = retriever.format_context(results)
# print("\n=== Context for LLM ===")
# print(context)
#
# # ۶. منابع
# sources = retriever.get_sources(results)
# print("\n=== Sources ===")
# for source in sources:
#     print(f"- {source}")

# tests/test_retriever.py

import pytest
import numpy as np
from core.retriever import Retriever
from core.embeddings import EmbeddingGenerator
from core.vector_store import FAISSVectorStore


@pytest.fixture
def retriever(sample_chunks):
    """Retriever با داده نمونه."""
    generator = EmbeddingGenerator()
    store = FAISSVectorStore(embedding_dim=generator.embedding_dimension)

    embeddings = generator.embed_chunks(sample_chunks)
    store.add_embeddings(sample_chunks, embeddings)

    return Retriever(generator, store, top_k=2, score_threshold=0.0)


def test_retrieve(retriever):
    """تست بازیابی."""
    results = retriever.retrieve("test query")

    assert len(results) <= 2
    assert all(hasattr(r, 'chunk') for r in results)
    assert all(hasattr(r, 'score') for r in results)


def test_format_context(retriever):
    """تست قالب‌بندی context."""
    results = retriever.retrieve("test query")
    context = retriever.format_context(results)

    assert isinstance(context, str)
    assert len(context) > 0


def test_get_sources(retriever):
    """تست استخراج منابع."""
    results = retriever.retrieve("test query")
    sources = retriever.get_sources(results)

    assert isinstance(sources, list)
    assert all(isinstance(s, str) for s in sources)
