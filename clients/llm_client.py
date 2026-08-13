"""Chat client for the Anthropic API, used by the RAGAS testset generator."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.anthropic.com")
API_KEY = os.environ.get("LLM_API_KEY", "LM_STUDIO")


chat_llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=BASE_URL,
    api_key=SecretStr(API_KEY),
)

async_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
