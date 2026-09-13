# core/chunker.py

"""
ماژول تقسیم متن به چانک‌های کوچک‌تر با overlap.
از sentence boundary detection برای فارسی و انگلیسی استفاده می‌کند.
"""

from dataclasses import dataclass
from typing import Protocol
import re

from core.loader import PageContent


@dataclass
class Chunk:
    """
    یک چانک متنی با متادیتای مربوطه.

    Attributes:
        text: متن چانک
        chunk_id: شناسه یکتای چانک (برای ردیابی)
        source_file: نام فایل منبع
        page_number: شماره صفحه منبع
        char_count: تعداد کاراکترها"""
    text: str
    chunk_id: str
    source_file: str
    page_number: int
    char_count: int

    @classmethod
    def from_text(
        cls,
        text: str,
        chunk_index: int,
        source_file: str,
        page_number: int
    ) -> "Chunk":
        """
        ساخت Chunk از متن خام.

        Args:
            text: متن چانک
            chunk_index: اندیس چانک در لیست
            source_file: نام فایل منبع
            page_number: شماره صفحه

        Returns:
            یک instance از Chunk
        """
        chunk_id = f"{source_file}_p{page_number}_c{chunk_index}"
        return cls(
            text=text.strip(),
            chunk_id=chunk_id,
            source_file=source_file,
            page_number=page_number,
            char_count=len(text.strip())
        )
class TextChunker:
    """
    تقسیم متن به چانک‌های کوچک‌تر با overlap.
    از sentence boundary detection برای حفظ یکپارچگی جملات استفاده می‌کند.

    Example:
        chunker = TextChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk_pages(pages)
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        """
        Args:
            chunk_size: حداکثر تعداد کاراکتر در هر چانک
            overlap: تعداد کاراکترهای مشترک بین چانک‌های متوالی
        """
        if overlap >= chunk_size:
            raise ValueError("overlap باید کوچک‌تر از chunk_size باشد")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: list[PageContent]) -> list[Chunk]:
        """
        تقسیم لیستی از صفحات به چانک‌ها.

        Args:
            pages: لیست PageContent از loader

        Returns:
            لیست Chunk‌های تولیدشده
        """
        all_chunks: list[Chunk] = []

        for page in pages:
            page_chunks = self._chunk_text(
                text=page.text,
                source_file=page.source_file,
                page_number=page.page_number
            )
            all_chunks.extend(page_chunks)

        return all_chunks

    def _chunk_text(
        self,
        text: str,
        source_file: str,
        page_number: int
    ) -> list[Chunk]:
        """
        تقسیم یک متن به چانک‌ها با رعایت sentence boundaries.

        Args:
            text: متن ورودی
            source_file: نام فایل منبع
            page_number: شماره صفحه

        Returns:
            لیست چانک‌های تولیدشده از این متن
        """
        # تقسیم به جملات
        sentences = self._split_sentences(text)

        chunks: list[Chunk] = []
        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            # اگر جمله فعلی + چانک فعلی بیشتر از حد مجاز شد
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # چانک فعلی رو ذخیره کن
                chunks.append(
                    Chunk.from_text(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        source_file=source_file,
                        page_number=page_number
                    )
                )
                chunk_index += 1

                # overlap رو حساب کن — آخرین N کاراکتر چانک قبلی
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                # جمله رو به چانک فعلی اضافه کن
                current_chunk += (" " if current_chunk else "") + sentence

        # چانک آخر رو اضافه کن
        if current_chunk.strip():
            chunks.append(
                Chunk.from_text(
                    text=current_chunk,
                    chunk_index=chunk_index,
                    source_file=source_file,
                    page_number=page_number
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        تقسیم متن به جملات — برای فارسی و انگلیسی.

        Args:
            text: متن ورودی

        Returns:
            لیست جملات
        """
        # الگوی regex برای تشخیص پایان جمله
        # نقطه، علامت سوال، علامت تعجب + فاصله یا خط جدید
        sentence_pattern = r'(?<=[.!?؟])\s+'

        sentences = re.split(sentence_pattern, text)

        # حذف جملات خالی
        return [s.strip() for s in sentences if s.strip()]

'''
چرا sentence-based؟

اگر وسط یه جمله ببریم، معنی از دست میره. با sentence boundary، هر چانک یه واحد معنایی کامل‌تره.

چرا overlap؟

فرض کن جمله مهم دقیقاً روی مرز دو چانک باشه — با overlap، هر دو چانک یه تیکه از اون جمله رو دارن → retrieval بهتر.

چرا regex ساده؟

برای نسخه اول کافیه. اگه بعداً خواستی دقیق‌تر بشه، می‌تونی hazm.sent_tokenize() برای فارسی و nltk.sent_tokenize() برای انگلیسی اضافه کنی.

چرا chunk_id داریم؟

وقتی retriever چانک رو برمی‌گردونه، می‌تونیم بگیم:

“این پاسخ از contract.pdf صفحه ۵، چانک ۳ اومده.”

Tradeoff:

چانک کوچیک‌تر = retrieval دقیق‌تر، ولی context کمتر برای LLM
چانک بزرگ‌تر = context بیشتر، ولی retrieval شلوغ‌تر
۵۰۰ کاراکتر (≈ ۱۰۰ کلمه) یه نقطه شروع خوبه.
'''