# core/retriever.py

"""
ماژول بازیابی context مرتبط از vector store.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore
from core.chunker import Chunk


@dataclass
class RetrievalResult:
    """
    نتیجه یک جستجو.

    Attributes:
        chunk: چانک بازیابی‌شده
        score: امتیاز شباهت (0-1)
        rank: رتبه در نتایج (1-based)
    """
    chunk: Chunk
    score: float
    rank: int

    def __str__(self) -> str:
        return (
            f"[Rank {self.rank} | Score: {self.score:.3f}]\n"
            f"Source: {self.chunk.source_file} (Page {self.chunk.page_number})\n"
            f"Text: {self.chunk.text[:200]}..."
        )


class Retriever:
    """
    بازیابی context مرتبط از vector store.

    Example:
        retriever = Retriever(generator, vector_store)
        results = retriever.retrieve("قرارداد چند ساله است؟", top_k=3)
        context = retriever.format_context(results)
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        score_threshold: float = 0.3
    ) -> None:
        """
        Args:
            embedding_generator: generator برای تبدیل query به embedding
            vector_store: vector store برای جستجو
            score_threshold: حداقل امتیاز برای فیلتر کردن نتایج ضعیف
        """
        self.generator = embedding_generator
        self.store = vector_store
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        apply_threshold: bool = True
    ) -> list[RetrievalResult]:
        """
        بازیابی مرتبط‌ترین چانک‌ها برای یک query.

        Args:
            query: سوال یا query کاربر
            top_k: تعداد نتایج برگشتی
            apply_threshold: اعمال score_threshold برای فیلتر کردن

        Returns:
            لیست RetrievalResult مرتب‌شده بر اساس score
        """
        if not query.strip():
            return []

        # ۱. تبدیل query به embedding
        query_embedding = self.generator.embed_text(query)

        # ۲. جستجو در vector store
        raw_results = self.store.search(query_embedding, top_k=top_k)

        # ۳. فیلتر کردن بر اساس threshold
        if apply_threshold:
            raw_results = [
                (chunk, score)
                for chunk, score in raw_results
                if score >= self.score_threshold
            ]

        # ۴. تبدیل به RetrievalResult
        results = [
            RetrievalResult(chunk=chunk, score=score, rank=i + 1)
            for i, (chunk, score) in enumerate(raw_results)
        ]

        return results

    def format_context(
        self,
        results: list[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """
        فرمت‌بندی نتایج به یک context متنی برای LLM.

        Args:
            results: لیست RetrievalResult
            include_metadata: آیا metadata (source, page) اضافه شود؟

        Returns:
            یک رشته متنی شامل تمام چانک‌ها
        """
        if not results:
            return "هیچ context مرتبطی یافت نشد."

        context_parts = []

        for result in results:
            if include_metadata:
                header = (
                    f"--- Context {result.rank} "
                    f"(Score: {result.score:.2f}, "
                    f"Source: {result.chunk.source_file}, "
                    f"Page: {result.chunk.page_number}) ---"
                )
                context_parts.append(header)

            context_parts.append(result.chunk.text)
            context_parts.append("")  # خط خالی بین چانک‌ها

        return "\n".join(context_parts)

    def get_sources(self, results: list[RetrievalResult]) -> list[str]:
        """
        استخراج لیست منابع یکتا از نتایج.

        Args:
            results: لیست RetrievalResult

        Returns:
            لیست نام فایل‌های منبع
        """
        sources = set()
        for result in results:
            source_info = f"{result.chunk.source_file} (Page {result.chunk.page_number})"
            sources.add(source_info)

        return sorted(list(sources))
def rerank_results(
    results: list[RetrievalResult],
    query: str,
    boost_recent: bool = False
) -> list[RetrievalResult]:
    """
    Re-ranking نتایج بر اساس معیارهای اضافی.

    Args:
        results: نتایج اولیه
        query: query اصلی
        boost_recent: آیا صفحات اولیه سند امتیاز بیشتری بگیرند؟

    Returns:
        نتایج re-rank شده
    """
    if not results:
        return results

    # مثال ساده: boost کردن چانک‌هایی که page_number کمتری دارند
    if boost_recent:
        for result in results:
            # صفحات اولیه (1-10) امتیاز بیشتر
            page_boost = 0.1 if result.chunk.page_number <= 10 else 0.0
            result.score += page_boost

    # مرتب‌سازی مجدد بر اساس score جدید
    results.sort(key=lambda r: r.score, reverse=True)

    # به‌روزرسانی rank
    for i, result in enumerate(results):
        result.rank = i + 1

    return results
