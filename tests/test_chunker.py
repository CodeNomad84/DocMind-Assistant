# from core.loader import PDFLoader
# from core.chunker import TextChunker
#
# # لود PDF
# loader = PDFLoader("data/raw/sample.pdf")
# pages = loader.load()
#
# # چانک کردن
# chunker = TextChunker(chunk_size=500, overlap=50)
# chunks = chunker.chunk_pages(pages)
#
# print(f"تعداد چانک‌ها: {len(chunks)}")
# print(f"اولین چانک:\n{chunks[0].text}")
# print(f"chunk_id: {chunks[0].chunk_id}")
#

# tests/test_chunker.py

import pytest
from core.chunker import TextChunker


def test_chunker_basic(sample_pages):
    """تست چانک کردن ساده."""
    chunker = TextChunker(chunk_size=30, overlap=5)
    chunks = chunker.chunk_pages(sample_pages)

    assert len(chunks) > 0
    assert all(chunk.char_count <= 30 for chunk in chunks)
    assert all(chunk.source_file == "test.pdf" for chunk in chunks)


def test_chunker_overlap(sample_pages):
    """تست همپوشانی."""
    chunker = TextChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_pages(sample_pages)

    # بررسی فرمت chunk_id
    assert all("_p" in chunk.chunk_id for chunk in chunks)
    assert all("_c" in chunk.chunk_id for chunk in chunks)


def test_chunker_empty_pages():
    """تست صفحات خالی."""
    from core.loader import PageContent

    empty_pages = [
        PageContent(1, "", "test.pdf", 0),
        PageContent(2, "   ", "test.pdf", 3)
    ]

    chunker = TextChunker()
    chunks = chunker.chunk_pages(empty_pages)

    assert len(chunks) == 0
