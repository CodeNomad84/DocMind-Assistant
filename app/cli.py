# app/cli.py

"""
رابط خط فرمان برای DocMind Assistant.
"""

import sys
from pathlib import Path

from app.config import get_settings
from core.loader import load_pdfs_from_directory
from core.chunker import TextChunker
from core.embeddings import EmbeddingGenerator
from core.vector_store import create_vector_store
from core.retriever import Retriever
from core.llm import OpenAIClient, GeminiClient
from core.qa_chain import QAChain


def index_documents():
    """ایندکس کردن تمام PDFهای موجود در data/raw."""
    settings = get_settings()

    print("🔄 شروع ایندکس‌سازی...")

    # ۱. بارگذاری PDFها
    print(f"📂 بارگذاری PDFها از {settings.raw_data_dir}...")
    pages = load_pdfs_from_directory(settings.raw_data_dir)

    if not pages:
        print("❌ هیچ PDFای یافت نشد!")
        return

    print(f"✅ {len(pages)} صفحه بارگذاری شد")

    # ۲. چانک کردن
    print("✂️  چانک کردن متن...")
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_pages(pages)
    print(f"✅ {len(chunks)} چانک ایجاد شد")

    # ۳. تولید embeddings
    print("🧠 تولید embeddings...")
    generator = EmbeddingGenerator(model_name=settings.embedding_model)
    embeddings = generator.embed_chunks(chunks)
    print(f"✅ {len(embeddings)} embedding تولید شد")

    # ۴. ذخیره در vector store
    print("💾 ذخیره در vector store...")
    store = create_vector_store(
        store_type=settings.vector_store_type,
        embedding_dim=generator.embedding_dimension
    )
    store.add_embeddings(chunks, embeddings)
    store.save(settings.vector_store_path)

    print(f"✅ ایندکس‌سازی کامل شد! ذخیره در {settings.vector_store_path}")


def query_documents(question: str):
    """پرسیدن سوال از اسناد ایندکس‌شده."""
    settings = get_settings()

    # بررسی وجود vector store
    if not Path(settings.vector_store_path).exists():
        print("❌ ابتدا باید اسناد را ایندکس کنید:")
        print("   python -m app.cli index")
        return

    print(f"🔍 جستجو برای: {question}\n")

    # ۱. بارگذاری vector store
    generator = EmbeddingGenerator(model_name=settings.embedding_model)
    store = create_vector_store(
        store_type=settings.vector_store_type,
        embedding_dim=generator.embedding_dimension
    )
    store.load(settings.vector_store_path)

    # ۲. ساخت retriever
    retriever = Retriever(generator, store)

    # ۳. ساخت LLM client
    if settings.llm_provider == "openai":
        llm = OpenAIClient(model=settings.llm_model)
    elif settings.llm_provider == "gemini":
        llm = GeminiClient(model=settings.llm_model)
    else:
        print(f"❌ LLM provider نامعتبر: {settings.llm_provider}")
        return

    # ۴. پرسیدن سوال
    chain = QAChain(retriever, llm, top_k=settings.top_k)
    result = chain.ask(question)

    # ۵. نمایش نتیجه
    print("📝 پاسخ:")
    print("-" * 80)
    print(result["answer"])
    print("-" * 80)

    if result["sources"]:
        print("\n📚 منابع:")
        for source in result["sources"]:
            print(f"  • {source}")

    if result["tokens_used"]:
        print(f"\n🔢 توکن مصرفی: {result['tokens_used']}")


def interactive_mode():
    """حالت تعاملی — پرسیدن چند سوال پشت سر هم."""
    settings = get_settings()

    if not Path(settings.vector_store_path).exists():
        print("❌ ابتدا باید اسناد را ایندکس کنید:")
        print("   python -m app.cli index")
        return

    print("🤖 DocMind Assistant — حالت تعاملی")
    print("برای خروج 'exit' یا 'quit' تایپ کنید\n")

    # بارگذاری یکبار
    generator = EmbeddingGenerator(model_name=settings.embedding_model)
    store = create_vector_store(
        store_type=settings.vector_store_type,
        embedding_dim=generator.embedding_dimension
    )
    store.load(settings.vector_store_path)

    retriever = Retriever(generator, store)

    if settings.llm_provider == "openai":
        llm = OpenAIClient(model=settings.llm_model)
    else:
        llm = GeminiClient(model=settings.llm_model)

    chain = QAChain(retriever, llm, top_k=settings.top_k)

    # حلقه تعاملی
    while True:
        try:
            question = input("\n❓ سوال شما: ").strip()

            if question.lower() in ["exit", "quit", "خروج"]:
                print("👋 خداحافظ!")
                break

            if not question:
                continue

            result = chain.ask(question)

            print("\n📝 پاسخ:")
            print(result["answer"])

            if result["sources"]:
                print("\n📚 منابع:", ", ".join(result["sources"]))
        except KeyboardInterrupt:
            print("\n\n👋 خداحافظ!")
            break
        except Exception as e:
            print(f"\n❌ خطا: {e}")


def main():
    """نقطه ورود CLI."""
    if len(sys.argv) < 2:
        print("استفاده:")
        print("  python -m app.cli index              # ایندکس کردن PDFها")
        print("  python -m app.cli query 'سوال'       # پرسیدن یک سوال")
        print("  python -m app.cli chat               # حالت تعاملی")
        return

    command = sys.argv[1]

    if command == "index":
        index_documents()
    elif command == "query":
        if len(sys.argv) < 3:
            print("❌ سوال را وارد کنید:")
            print("   python -m app.cli query 'قرارداد چند ساله است?'")
            return
        question = " ".join(sys.argv[2:])
        query_documents(question)

    elif command == "chat":
        interactive_mode()

    else:
        print(f"❌ دستور نامعتبر: {command}")
        print("دستورات معتبر: index, query, chat")


if __name__ == "__main__":
    main()
