# core/qa_chain.py

"""
زنجیره کامل Question-Answering.
"""

from core.retriever import Retriever
from core.llm import LLMClient, LLMResponse


class QAChain:
    """
    زنجیره کامل: Query → Retrieval → LLM → Answer

    Example:
        chain = QAChain(retriever, llm_client)
        result = chain.ask("قرارداد چند ساله است؟")print(result.answer)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        top_k: int = 5
    ) -> None:
        """
        Args:
            retriever: Retriever برای بازیابی context
            llm_client: LLM client برای تولید پاسخ
            top_k: تعداد چانک‌های بازیابی‌شده
        """
        self.retriever = retriever
        self.llm = llm_client
        self.top_k = top_k

    def ask(
        self,
        query: str,
        system_prompt: str | None = None,
        return_sources: bool = True
    ) -> dict:
        """
        پرسیدن سوال و دریافت پاسخ.

        Args:
            query: سوال کاربر
            system_prompt: دستورالعمل سیستم (اختیاری)
            return_sources: آیا منابع برگردونده بشه؟

        Returns:
            dict شامل answer, sources, tokens_used
        """
        # ۱. بازیابی context
        results = self.retriever.retrieve(query, top_k=self.top_k)

        if not results:
            return {
                "answer": "هیچ اطلاعات مرتبطی در اسناد یافت نشد.",
                "sources": [],
                "tokens_used": 0
            }

        # ۲. فرمت context
        context = self.retriever.format_context(results, include_metadata=False)

        # ۳. تولید پاسخ
        response = self.llm.generate(query, context, system_prompt)

        # ۴. استخراج منابع
        sources = self.retriever.get_sources(results) if return_sources else []

        return {
            "answer": response.answer,
            "sources": sources,
            "tokens_used": response.tokens_used
        }
