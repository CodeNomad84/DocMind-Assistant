# tests/conftest.py

import pytest
from pathlib import Path
import tempfile
import shutil

from core.loader import PageContent
from core.chunker import Chunk


@pytest.fixture
def temp_dir():
    """پوشه موقت."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_pages():
    """صفحات نمونه."""
    return [
        PageContent(
            page_number=1,
            text="This is the first page. It contains some text.",
            source_file="test.pdf",
            char_count=48
        ),
        PageContent(
            page_number=2,
            text="این صفحه دوم است. شامل متن فارسی می‌باشد.",
            source_file="test.pdf",
            char_count=45
        )
    ]


@pytest.fixture
def sample_chunks():
    """چانک‌های نمونه."""
    return [
        Chunk(
            text="First chunk of text",
            chunk_id="test.pdf_p1_c0",
            source_file="test.pdf",
            page_number=1,
            char_count=19
        ),
        Chunk(
            text="Second chunk of text",
            chunk_id="test.pdf_p1_c1",
            source_file="test.pdf",
            page_number=1,
            char_count=20
        )
    ]
