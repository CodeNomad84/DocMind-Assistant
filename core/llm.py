# core/llm.py

"""
ماژول ارتباط با LLM برای تولید پاسخ.
"""

from dataclasses import dataclass
from typing import Protocol
import os

from openai import OpenAI


@dataclass
class LLMResponse:
    """
    پاسخ LLM.

    Attributes:
        answer: پاسخ نهایی
        model: نام مدل استفاده‌شده
        tokens_used: تعداد توکن مصرف‌شده (اختیاری)
    """
    answer: str
    model: str
    tokens_used: int | None = None


class LLMClient(Protocol):
    """Interface برای LLM clients مختلف."""

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """تولید پاسخ از LLM."""
        ...


class OpenAIClient:
    """
    Client برای OpenAI API.

    Example:
        client = OpenAIClient(model="gpt-4o-mini")
        response = client.generate(
            query="قرارداد چند ساله است؟",
            context="مدت قرارداد ۳ سال است..."
        )
    """

    DEFAULT_SYSTEM_PROMPT = """تو یک دستیار هوشمند هستی که به سوالات کاربر بر اساس context داده‌شده پاسخ می‌دهی.

قوانین:
- فقط از اطلاعات موجود در context استفاده کن
- اگر پاسخ در context نیست، بگو "اطلاعات کافی در اسناد موجود نیست"
- پاسخ‌ها رو واضح و مختصر بده
- اگر منبع مشخصی داری، اشاره کن"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.1
    ) -> None:
        """
        Args:
            model: نام مدل OpenAI
            api_key: کلید API (اگر None باشه از env می‌خونه)
            temperature: میزان خلاقیت (0-1)
        """
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """
        تولید پاسخ از OpenAI.

        Args:
            query: سوال کاربر
            context: context بازیابی‌شده از retriever
            system_prompt: دستورالعمل سیستم (اختیاری)

        Returns:
            LLMResponse شامل پاسخ
        """
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        user_message = f"""Context:
{context}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature
        )

        return LLMResponse(
            answer=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else None
        )


class GeminiClient:
    """
    Client برای Google Gemini API.

    Example:
        client = GeminiClient(model="gemini-2.0-flash-exp")
        response = client.generate(query, context)
    """

    DEFAULT_SYSTEM_PROMPT = OpenAIClient.DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        temperature: float = 0.1
    ) -> None:
        """
        Args:
            model: نام مدل Gemini
            api_key: کلید API
            temperature: میزان خلاقیت
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("نصب کن: pip install google-generativeai")

        self.model_name = model
        self.temperature = temperature

        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """تولید پاسخ از Gemini."""
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""

        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )

        return LLMResponse(
            answer=response.text,
            model=self.model_name,
            tokens_used=None  # Gemini API فعلاً token count نمی‌ده
        )
