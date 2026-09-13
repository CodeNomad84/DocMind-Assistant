# app/config.py

"""
مرکز تنظیمات پروژه DocMind Assistant.
تمام ماژول‌ها تنظیمات خود را از این فایل می‌خوانند.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ریشه پروژه — دو سطح بالاتر از این فایل
PROJECT_ROOT = Path(__file__).parent.parent


class get_settings(BaseSettings):
    """
    تنظیمات کلی برنامه با استفاده از Pydantic BaseSettings.
    مقادیر به‌صورت خودکار از فایل .env خوانده می‌شوند.
    """

    # --- API Keys ---
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")

    # --- LLM Settings ---
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")

    # --- Embedding Settings ---
    embedding_model: str = Field(
        default="paraphrase-multilingual-mpnet-base-v2",
        env="EMBEDDING_MODEL"
    )

    # --- Vector Store ---
    vector_store_type: str = Field(default="faiss", env="VECTOR_STORE_TYPE") # می شود از هم استفاده کرد "chroma"

    # --- Chunking ---
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")

    # --- Paths (همیشه نسبت به ریشه پروژه) ---
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "ignore"}



# یک instance واحد برای کل پروژه (Singleton pattern)
settings = get_settings()


def ensure_directories() -> None:
    """
    پوشه‌های مورد نیاز را در صورت نبود می‌سازد.
    این تابع باید در ابتدای اجرای برنامه فراخوانی شود.
    """
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Directories are ready.")

