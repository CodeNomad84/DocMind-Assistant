# core/vector_store.py

"""
ماژول مدیریت vector store برای ذخیره و جستجوی embeddingها.
پشتیبانی از FAISS و ChromaDB.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol
import pickle

import numpy as np
import faiss

from core.chunker import Chunk
from app.config import settings


class VectorStore(ABC):
    """
    کلاس پایه برای vector storeها.
    """

    @abstractmethod
    def add_embeddings(
            self,
            chunks: list[Chunk],
            embeddings: list[np.ndarray]
    ) -> None:
        """افزودن embeddingها به store."""
        pass

    @abstractmethod
    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        جستجوی نزدیک‌ترین چانک‌ها به query.

        Returns:
            لیست تاپل‌های (Chunk, similarity_score)
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """ذخیره vector store در دیسک."""
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """بارگذاری vector store از دیسک."""
        pass


# ============================================================
# گزینه ۱: FAISS
# ============================================================

class FAISSVectorStore(VectorStore):
    """
    Vector store با استفاده از FAISS.

    مزایا:
    - سرعت بسیار بالا
    - مصرف حافظه کم
    - مناسب برای میلیون‌ها vector

    Example:
        store = FAISSVectorStore(embedding_dim=768)
        store.add_embeddings(chunks, embeddings)
        results = store.search(query_embedding, top_k=3)
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        Args:
            embedding_dim: ابعاد embedding (مثلاً 768)
        """
        self.embedding_dim = embedding_dim

        # ایجاد FAISS index — IndexFlatL2 برای L2 distance
        self.index = faiss.IndexFlatL2(embedding_dim)

        # ذخیره چانک‌ها برای بازیابی بعدی
        self.chunks: list[Chunk] = []

    def add_embeddings(
            self,
            chunks: list[Chunk],
            embeddings: list[np.ndarray]
    ) -> None:
        """
        افزودن embeddingها به FAISS index.

        Args:
            chunks: لیست Chunk‌ها
            embeddings: لیست embeddingها (numpy arrays)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("تعداد chunks و embeddings باید برابر باشد")

        if not embeddings:
            return

        # تبدیل لیست embeddingها به یک آرایه 2D
        embeddings_array = np.array(embeddings).astype('float32')

        # افزودن به index
        self.index.add(embeddings_array)

        # ذخیره چانک‌ها
        self.chunks.extend(chunks)

        print(f"✅ {len(chunks)} embedding به FAISS اضافه شد. کل: {self.index.ntotal}")

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        جستجوی نزدیک‌ترین چانک‌ها.

        Args:
            query_embedding: embedding سوال
            top_k: تعداد نتایج برگشتی

        Returns:
            لیست (Chunk, distance) — distance کمتر = شباهت بیشتر
        """
        if self.index.ntotal == 0:
            return []

        # تبدیل به فرمت مناسب FAISS
        query_array = np.array([query_embedding]).astype('float32')

        # جستجو
        distances, indices = self.index.search(query_array, top_k)

        # ساخت نتایج
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # چک کردن index معتبر
                # تبدیل distance به similarity score (0-1)
                # distance کمتر = similarity بیشتر
                similarity = 1 / (1 + distance)
                results.append((self.chunks[idx], similarity))

        return results

    def save(self, path: str | Path) -> None:
        """
        ذخیره FAISS index و چانک‌ها.

        Args:
            path: مسیر دایرکتوری برای ذخیره
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # ذخیره FAISS index
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # ذخیره چانک‌ها
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"✅ FAISS vector store در {path} ذخیره شد.")

    def load(self, path: str | Path) -> None:
        """
        بارگذاری FAISS index و چانک‌ها.

        Args:
            path: مسیر دایرکتوری
        """
        path = Path(path)

        # بارگذاری FAISS index
        index_path = path / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"فایل index یافت نشد: {index_path}")

        self.index = faiss.read_index(str(index_path))

        # بارگذاری چانک‌ها
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✅ FAISS vector store بارگذاری شد. تعداد vectors: {self.index.ntotal}")


# ============================================================
# گزینه ۲: ChromaDB
# ============================================================

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ChromaVectorStore(VectorStore):
    """
    Vector store با استفاده از ChromaDB.

    مزایا:
    - API ساده‌تر
    - metadata filtering قوی‌تر
    - مناسب برای پروژه‌های کوچک تا متوسط

    Example:
        store = ChromaVectorStore(collection_name="documents")
        store.add_embeddings(chunks, embeddings)
        results = store.search(query_embedding, top_k=3)
    """

    def __init__(
            self,
            collection_name: str = "documents",
            persist_directory: str | Path | None = None
    ) -> None:
        """
        Args:
            collection_name: نام collection در ChromaDB
            persist_directory: مسیر ذخیره‌سازی (اگر None باشد، از config استفاده می‌شود)
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB نصب نیست. برای نصب:\n"
                "pip install chromadb"
            )

        persist_dir = persist_directory or settings.data_dir / "chroma"
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # ایجاد client
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # ایجاد یا بارگذاری collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # استفاده از cosine similarity
        )

        print(f"✅ ChromaDB collection '{collection_name}' آماده است.")

    def add_embeddings(
            self,
            chunks: list[Chunk],
            embeddings: list[np.ndarray]
    ) -> None:
        """
        افزودن embeddingها به ChromaDB.

        Args:
            chunks: لیست Chunk‌ها
            embeddings: لیست embeddingها
        """
        if len(chunks) != len(embeddings):
            raise ValueError("تعداد chunks و embeddings باید برابر باشد")

        if not chunks:
            return

        # آماده‌سازی داده‌ها برای ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "char_count": chunk.char_count
            }
            for chunk in chunks
        ]
        embeddings_list = [emb.tolist() for emb in embeddings]

        # افزودن به collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

        print(f"✅ {len(chunks)} embedding به ChromaDB اضافه شد.")

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        جستجوی نزدیک‌ترین چانک‌ها.

        Args:
            query_embedding: embedding سوال
            top_k: تعداد نتایج

        Returns:
            لیست (Chunk, similarity_score)
        """
        # جستجو در collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # تبدیل نتایج به فرمت استاندارد
        output = []
        for i in range(len(results['ids'][0])):
            chunk = Chunk(
                text=results['documents'][0][i],
                chunk_id=results['ids'][0][i],
                source_file=results['metadatas'][0][i]['source_file'],
                page_number=results['metadatas'][0][i]['page_number'],
                char_count=results['metadatas'][0][i]['char_count']
            )
            # ChromaDB برمی‌گرداند distance — تبدیل به similarity
            distance = results['distances'][0][i]
            similarity = 1 - distance  # cosine distance → similarity

            output.append((chunk, similarity))

        return output

    def save(self, path: str | Path) -> None:
        """
        ChromaDB به‌صورت خودکار persist می‌شود.
        این متد برای سازگاری با interface وجود دارد.
        """
        print("✅ ChromaDB به‌صورت خودکار ذخیره می‌شود.")

        def load(self, path: str | Path) -> None:
            """
            ChromaDB به‌صورت خودکار load می‌شود.
            این متد برای سازگاری با interface وجود دارد.
            """

        print("✅ ChromaDB به‌صورت خودکار بارگذاری می‌شود.")


# ============================================================
# Factory Function
# ============================================================

def create_vector_store(
        store_type: str = "faiss",
        embedding_dim: int = 768,
        **kwargs
) -> VectorStore:
    """
    ساخت vector store بر اساس نوع.

    Args:
        store_type: "faiss" یا "chroma"
        embedding_dim: ابعاد embedding (فقط برای FAISS)
        **kwargs: آرگومان‌های اضافی برای هر store

    Returns:
        یک instance از VectorStore

    Example:
        store = create_vector_store("faiss", embedding_dim=768)
    """
    if store_type.lower() == "faiss":
        return FAISSVectorStore(embedding_dim=embedding_dim)

    elif store_type.lower() == "chroma":
        return ChromaVectorStore(**kwargs)

    else:
        raise ValueError(f"نوع vector store پشتیبانی نمی‌شود: {store_type}")
