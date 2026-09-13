# from core.loader import PDFLoader
# from core.chunker import TextChunker
# from core.embeddings import EmbeddingGenerator, save_embeddings
#
# # Pipeline کامل تا اینجا
# loader = PDFLoader("data/raw/sample.pdf")
# pages = loader.load()
#
# chunker = TextChunker(chunk_size=500, overlap=50)
# chunks = chunker.chunk_pages(pages)
#
# generator = EmbeddingGenerator()
# embeddings = generator.embed_chunks(chunks)
#
# # ذخیره
# save_embeddings(chunks, embeddings, "data/processed/embeddings.pkl")
#
# print(f"شکل اولین embedding: {embeddings[0].shape}")

# tests/test_embeddings.py

import pytest
import numpy as np
from core.embeddings import EmbeddingGenerator


@pytest.fixture
def generator():
    """تولیدکننده embedding."""
    return EmbeddingGenerator(model_name="paraphrase-multilingual-mpnet-base-v2")


def test_embed_text(generator):
    """تست embedding تکی."""
    text = "This is a test sentence."
    embedding = generator.embed_text(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == generator.embedding_dimension
    assert not np.isnan(embedding).any()


def test_embed_chunks(generator, sample_chunks):
    """تست embedding دسته‌ای."""
    embeddings = generator.embed_chunks(sample_chunks)

    assert len(embeddings) == len(sample_chunks)
    assert all(emb.shape[0] == generator.embedding_dimension for emb in embeddings)


def test_embed_multilingual(generator):
    """تست چندزبانه."""
    texts = [
        "English text",
        "متن فارسی",
        "Mixed متن"
    ]

    embeddings = [generator.embed_text(text) for text in texts]

    assert all(emb.shape[0] == generator.embedding_dimension for emb in embeddings)
