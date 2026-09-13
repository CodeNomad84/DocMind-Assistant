# core/loader.py

"""
ماژول بارگذاری و استخراج متن از فایل‌های PDF.
از PyMuPDF (fitz) برای استخراج استفاده می‌شود.
"""

from dataclasses import dataclass, field
from pathlib import Path
import fitz  # PyMuPDF


@dataclass
class PageContent:
    """
    محتوای استخراج‌شده از یک صفحه PDF.

    Attributes:
        page_number: شماره صفحه (از ۱ شروع می‌شود)
        text: متن خام استخراج‌شده
        source_file: نام فایل منبع
        char_count: تعداد کاراکترهای متن
    """
    page_number: int
    text: str
    source_file: str
    char_count: int = field(init=False)

    def __post_init__(self):
        # char_count رو خودکار محاسبه می‌کنیم
        self.char_count = len(self.text)

    def is_empty(self) -> bool:
        """برگرداندن True اگر صفحه متن قابل استفاده‌ای نداشته باشد."""
        return len(self.text.strip()) < 10

class PDFLoader:
    """
    بارگذاری و استخراج متن از فایل‌های PDF.

    Example:
        loader = PDFLoader("data/raw/document.pdf")
        pages = loader.load()
        for page in pages:
            print(page.page_number, page.text[:100])
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Args:
            file_path: مسیر فایل PDF
        
        Raises:
            FileNotFoundError: اگر فایل وجود نداشته باشد
            ValueError: اگر فایل PDF نباشد
        """
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self) -> None:
        """اعتبارسنجی وجود و نوع فایل."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"فایل یافت نشد: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"فایل باید PDF باشد: {self.file_path.suffix}")

    def load(self) -> list[PageContent]:
        """
        استخراج متن از تمام صفحات PDF.

        Returns:
            لیستی از PageContent برای هر صفحه غیر خالی

        Raises:
            RuntimeError: اگر فایل PDF خراب یا رمزگذاری‌شده باشد
        """
        pages: list[PageContent] = []

        try:
            # باز کردن فایل PDF
            doc = fitz.open(str(self.file_path))
        except Exception as e:
            raise RuntimeError(f"خطا در باز کردن PDF: {e}") from e

        with doc:  # با context manager مطمئن می‌شیم فایل بسته می‌شه
            for page_index in range(len(doc)):
                page = doc[page_index]

                # استخراج متن — "text" ساده‌ترین حالته
                raw_text = page.get_text("text")

                page_content = PageContent(
                    page_number=page_index + 1,  # از ۱ شروع می‌شه نه ۰
                    text=self._clean_text(raw_text),
                    source_file=self.file_path.name,
                )

                # صفحات خالی رو رد می‌کنیم
                if not page_content.is_empty():
                    pages.append(page_content)

        return pages

    def _clean_text(self, text: str) -> str:
        """
        پاک‌سازی اولیه متن استخراج‌شده.
        
        Args:
            text: متن خام از PyMuPDF

        Returns:
            متن پاک‌شده
        """
        # حذف خطوط خالی اضافه
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned

def load_pdfs_from_directory(directory: str | Path) -> list[PageContent]:
    """
    بارگذاری تمام فایل‌های PDF از یک پوشه.

    Args:
        directory: مسیر پوشه حاوی فایل‌های PDF

    Returns:
        لیست ترکیبی از تمام صفحات همه فایل‌ها
    """
    directory = Path(directory)
    all_pages: list[PageContent] = []

    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  هیچ فایل PDF‌ای در {directory} یافت نشد.")
        return all_pages

    for pdf_file in pdf_files:
        try:
            loader = PDFLoader(pdf_file)
            pages = loader.load()
            all_pages.extend(pages)
            print(f"✅ {pdf_file.name}: {len(pages)} صفحه بارگذاری شد.")
        except Exception as e:
            # یه فایل خراب نباید کل فرآیند رو متوقف کنه
            print(f"❌ خطا در {pdf_file.name}: {e}")

    return all_pages
