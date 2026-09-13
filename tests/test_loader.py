# from core.loader import PDFLoader
#
# loader = PDFLoader("data/raw/sample.pdf")
# pages = loader.load()
#
# print(len(pages))
# print(pages[0].text[:500])

# tests/test_loader.py

import pytest
from pathlib import Path
from core.loader import PDFLoader, PageContent


def test_pdf_loader_validation():
    """تست اعتبارسنجی فایل."""
    with pytest.raises(FileNotFoundError):
        PDFLoader("nonexistent.pdf")

    with pytest.raises(ValueError):
        PDFLoader("test.txt")


def test_page_content_creation():
    """تست ساخت PageContent."""
    page = PageContent(
        page_number=1,
        text="Sample text",
        source_file="test.pdf",
        char_count=11
    )

    assert page.page_number == 1
    assert page.text == "Sample text"
    assert page.char_count == 11
