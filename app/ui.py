# app/ui.py

"""
رابط گرافیکی Streamlit برای DocMind Assistant.
"""

import streamlit as st
from pathlib import Path
import shutil

from app.config import get_settings
from core.loader import PDFLoader
from core.chunker import TextChunker
from core.embeddings import EmbeddingGenerator
from core.vector_store import create_vector_store
from core.retriever import Retriever
from core.llm import OpenAIClient, GeminiClient
from core.qa_chain import QAChain

# تنظیمات صفحه
st.set_page_config(
    page_title="DocMind Assistant",
    page_icon="🤖",
    layout="wide"
)


def init_session_state():
    """مقداردهی اولیه session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "indexed" not in st.session_state:
        settings = get_settings()
        st.session_state.indexed = Path(settings.vector_store_path).exists()


def index_page():
    """صفحه ایندکس کردن PDFها."""
    st.title("📚 ایندکس اسناد")

    settings = get_settings()

    # آپلود فایل
    uploaded_files = st.file_uploader(
        "فایل‌های PDF را آپلود کنید",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🔄 ایندکس کردن", type="primary", disabled=not uploaded_files):
        if not uploaded_files:
            st.warning("لطفاً حداقل یک فایل PDF آپلود کنید")
            return

        progress_bar = st.progress(0)
        status = st.empty()

        try:
            # ۱. ذخیره فایل‌ها
            status.text("💾 ذخیره فایل‌ها...")
            settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = settings.raw_data_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            progress_bar.progress(20)

            # ۲. بارگذاری و چانک کردن
            status.text("📂 بارگذاری PDFها...")
            all_pages = []
            for uploaded_file in uploaded_files:
                file_path = settings.raw_data_dir / uploaded_file.name
                loader = PDFLoader(str(file_path))
                pages = loader.load()
                all_pages.extend(pages)

            progress_bar.progress(40)

            status.text("✂️ چانک کردن متن...")
            chunker = TextChunker(
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap
            )
            chunks = chunker.chunk_pages(all_pages)

            progress_bar.progress(60)

            # ۳. تولید embeddings
            status.text("🧠 تولید embeddings...")
            generator = EmbeddingGenerator(model_name=settings.embedding_model)
            embeddings = generator.embed_chunks(chunks)

            progress_bar.progress(80)

            # ۴. ذخیره در vector store
            status.text("💾 ذخیره در vector store...")
            store = create_vector_store(
                store_type=settings.vector_store_type,
                embedding_dim=generator.embedding_dimension
            )
            store.add_embeddings(chunks, embeddings)
            store.save(settings.vector_store_path)

            progress_bar.progress(100)
            status.empty()

            st.success(f"✅ {len(chunks)} چانک از {len(uploaded_files)} فایل ایندکس شد!")
            st.session_state.indexed = True
            st.session_state.qa_chain = None  # Reset chain

        except Exception as e:
            st.error(f"❌ خطا: {e}")


def chat_page():
    """صفحه چت."""
    st.title("💬 پرسش و پاسخ")

    settings = get_settings()

    # بررسی ایندکس
    if not st.session_state.indexed:
        st.warning("⚠️ ابتدا باید اسناد را ایندکس کنید")
        st.page_link("index", label="رفتن به صفحه ایندکس", icon="📚")
        return

    # بارگذاری QA chain (یکبار)
    if st.session_state.qa_chain is None:
        with st.spinner("🔄 بارگذاری مدل..."):
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

            st.session_state.qa_chain = QAChain(retriever, llm, top_k=settings.top_k)
    # نمایش تاریخچه
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 منابع"):
                    for source in message["sources"]:
                        st.text(f"• {source}")

    # ورودی کاربر
    if prompt := st.chat_input("سوال خود را بپرسید..."):
        # نمایش سوال
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # تولید پاسخ
        with st.chat_message("assistant"):
            with st.spinner("🤔 در حال فکر کردن..."):
                result = st.session_state.qa_chain.ask(prompt)

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📚 منابع"):
                        for source in result["sources"]:
                            st.text(f"• {source}")

                # ذخیره در تاریخچه
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

    # دکمه پاک کردن تاریخچه
    if st.session_state.messages:
        if st.button("🗑️ پاک کردن تاریخچه"):
            st.session_state.messages = []
            st.rerun()


def sidebar():
    """نوار کناری."""
    with st.sidebar:
        st.title("🤖 DocMind Assistant")
        st.markdown("---")

        settings = get_settings()

        # وضعیت
        if st.session_state.indexed:
            st.success("✅ اسناد ایندکس شده")
        else:
            st.warning("⚠️ اسناد ایندکس نشده")

        st.markdown("---")

        # تنظیمات
        st.subheader("⚙️ تنظیمات")
        st.text(f"LLM: {settings.llm_provider}")
        st.text(f"Model: {settings.llm_model}")
        st.text(f"Top-K: {settings.top_k}")

        st.markdown("---")

        # راهنما
        with st.expander("📖 راهنما"):
            st.markdown("""
            **نحوه استفاده:**
            1. در صفحه **ایندکس**، PDFها را آپلود کنید
            2. روی **ایندکس کردن** کلیک کنید
            3. به صفحه **چت** بروید
            4. سوال خود را بپرسید
            """)
    # تنظیمات پیشرفته در sidebar
    with st.sidebar.expander("🔧 تنظیمات پیشرفته"):
        top_k = st.slider("تعداد نتایج", 1, 10, settings.top_k)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)

    # نمایش token usage
    if result["tokens_used"]:
        st.caption(f"🔢 توکن مصرفی: {result['tokens_used']}")

    # دانلود تاریخچه
    if st.session_state.messages:
        history_text = "\n\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages
        ])
        st.download_button(
            "💾 دانلود تاریخچه",
            history_text,
            "chat_history.txt"
        )


def main():
    """نقطه ورود اصلی."""
    init_session_state()
    sidebar()

    # Navigation
    page = st.navigation([
        st.Page(index_page, title="ایندکس", icon="📚"),
        st.Page(chat_page, title="چت", icon="💬")
    ])

    page.run()


if __name__ == "__main__":
    m۳.اضافه
    کردن
    به
    `requirements.txt`
```txt
streamlit >= 1.32
.0
