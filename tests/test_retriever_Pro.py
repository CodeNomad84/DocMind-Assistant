# استفاده پیشرفته با re-ranking

from core.retriever import Retriever, rerank_results

# جستجو
results = retriever.retrieve(query, top_k=10)

# re-ranking
results = rerank_results(
    results,
    query=query,
    boost_recent=True  # صفحات اولیه امتیاز بیشتر
)

# انتخاب top 5 بعد از re-ranking
results = results[:5]

# فرمت context
context = retriever.format_context(results)
