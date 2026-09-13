# core/embeddings.py

"""
ماژول تولید embedding از متن.
از Sentence Transformers برای embedding محلی استفاده می‌کند.
"""

from typing import Protocol
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from core.chunker import Chunk


class EmbeddingGenerator:
    """
    تولید embedding از متن با استفاده از Sentence Transformers.

    مدل پیش‌فرض: paraphrase-multilingual-mpnet-base-v2
    این مدل از فارسی و انگلیسی پشتیبانی می‌کند.

    Example:
        generator = EmbeddingGenerator()
        embedding = generator.embed_text("سلام دنیا")
        print(embedding.shape)  # (768,)
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Args:
            model_name: نام مدل Sentence Transformers.
                       اگر None باشد، از تنظیمات config استفاده می‌شود.
        """
        self.model_name = model_name or settings.embedding_model
        print(f"🔄 در حال بارگذاری مدل embedding: {self.model_name}")

        # بارگذاری مدل — اولین بار کمی طول می‌کشد
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ مدل بارگذاری شد. ابعاد embedding: {self.model.get_sentence_embedding_dimension()}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        تولید embedding از یک متن.

        Args:
            text: متن ورودی

        Returns:
            آرایه numpy با ابعاد (embedding_dim,)
        """
        if not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")

        # encode برمی‌گرداند: numpy array با shape (embedding_dim,)
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embedding

    def embed_chunks(self, chunks: list[Chunk]) -> list[np.ndarray]:
        """
        تولید embedding از لیستی از چانک‌ها به‌صورت batch.

        Args:
            chunks: لیست Chunk‌ها

        Returns:
            لیست numpy arrayها — هر کدام یک embedding
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        print(f"🔄 در حال تولید embedding برای {len(texts)} چانک...")

        # batch encoding — سریع‌تر از تک‌تک
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32  # تنظیم بر اساس RAM
        )

        print(f"✅ {len(embeddings)} embedding تولید شد.")

        return list(embeddings)

    @property
    def embedding_dimension(self) -> int:
        """برگرداندن ابعاد embedding مدل."""
        return self.model.get_sentence_embedding_dimension()

from pathlib import Path
import pickle


def save_embeddings(
    chunks: list[Chunk],
    embeddings: list[np.ndarray],
    output_path: str | Path
) -> None:
    """
    ذخیره چانک‌ها و embeddingهای آن‌ها در یک فایل pickle.

    Args:
        chunks: لیست Chunk‌ها
        embeddings: لیست embeddingها (باید هم‌اندازه با chunks باشد)
        output_path: مسیر فایل خروجی
    """
    if len(chunks) != len(embeddings):
        raise ValueError("تعداد chunks و embeddings باید برابر باشد")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "chunks": chunks,
        "embeddings": embeddings,
        "model_name": settings.embedding_model,
        "embedding_dim": embeddings[0].shape[0] if embeddings else 0
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ {len(chunks)} چانک و embedding در {output_path} ذخیره شد.")


def load_embeddings(input_path: str | Path) -> tuple[list[Chunk], list[np.ndarray]]:
    """
    بارگذاری چانک‌ها و embeddingها از فایل pickle.

    Args:
        input_path: مسیر فایل ورودی

    Returns:
        تاپل (chunks, embeddings)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"فایل یافت نشد: {input_path}")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"✅ {len(data['chunks'])} چانک و embedding بارگذاری شد.")
    print(f"   مدل: {data['model_name']}, ابعاد: {data['embedding_dim']}")

    return data["chunks"], data["embeddings"]
