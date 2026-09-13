# # استفاده پایه
# from core.embeddings import EmbeddingGenerator
# from core.vector_store import FAISSVectorStore
# from core.retriever import Retriever
# from core.llm import OpenAIClient
# from core.qa_chain import QAChain
#
# # ۱. بارگذاری vector store
# generator = EmbeddingGenerator()
# store = FAISSVectorStore(embedding_dim=768)
# store.load("data/processed/vector_store")
#
# # ۲. ساخت retriever
# retriever = Retriever(generator, store)
#
# # ۳. ساخت LLM client
# llm = OpenAIClient(model="gpt-4o-mini")
#
# # ۴. ساخت QA chain
# chain = QAChain(retriever, llm, top_k=5)
#
# # ۵. پرسیدن سوال
# result = chain.ask("قرارداد چند ساله است؟")
#
# print("پاسخ:", result["answer"])
# print("\nمنابع:")
# for source in result["sources"]:
#     print(f"  - {source}")
# print(f"\nتوکن مصرفی: {result['tokens_used']}")
#
# # استفاده با Gemini
# from core.llm import GeminiClient
#
# llm = GeminiClient(model="gemini-2.0-flash-exp")
# chain = QAChain(retriever, llm)
#
# result = chain.ask("مبلغ قرارداد چقدر است؟")
# print(result["answer"])
#
# # Custom System Prompt
# custom_prompt = """تو یک وکیل حرفه‌ای هستی.
# به سوالات حقوقی بر اساس اسناد پاسخ بده.
# اگر اطلاعات کافی نیست، توصیه کن که با وکیل مشورت کنه."""
#
# result = chain.ask(
#     "آیا می‌توانم قرارداد را فسخ کنم؟",
#     system_prompt=custom_prompt
# )

# tests/test_qa_chain.py

import pytest
from unittest.mock import Mock
from core.qa_chain import QAChain


@pytest.fixture
def mock_retriever():
    """Mock retriever."""
    retriever = Mock()
    retriever.retrieve.return_value = []
    retriever.format_context.return_value = "Mock context"
    retriever.get_sources.return_value = ["test.pdf"]
    return retriever


@pytest.fixture
def mock_llm():
    """Mock LLM."""
    llm = Mock()
    llm.generate.return_value = Mock(
        response="Mock answer",
        model="mock-model",
        tokens_used=100
    )
    return llm


def test_qa_chain_ask(mock_retriever, mock_llm):
    """تست پرسیدن سوال."""
    chain = QAChain(mock_retriever, mock_llm, top_k=3)
    result = chain.ask("What is this?")

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "Mock answer"
    assert result["sources"] == ["test.pdf"]

    mock_retriever.retrieve.assert_called_once()
    mock_llm.generate.assert_called_once()
