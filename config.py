"""
Project configuration defaults. Values are read from environment variables
so you can override them without editing this file (or place them in a
.env file if you use a loader).

Add or tweak values as needed for your environment (API keys, model names,
paths, thresholds, etc.).
"""
from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root into os.environ, if present


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


# Embedding / dataset / intent model
EMBED_MODEL: str = _env("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
BITEXT_DATASET_PATH: str = _env(
    "BITEXT_DATASET_PATH",
    "hf://datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/bitext-telco-llm-chatbot-training-dataset.csv",
)
INTENT_MODEL_PATH: str = _env("INTENT_MODEL_PATH", "mobilis_intent_model")


# Classifier thresholds and OOD keywords
CONFIDENCE_THRESHOLD: float = float(_env("CONFIDENCE_THRESHOLD", "0.5"))
MARGIN_THRESHOLD: float = float(_env("MARGIN_THRESHOLD", "0.15"))
OUT_OF_SCOPE_KEYWORDS: Dict[str, str] = {
    "technician": "human_agent",
    "technician visit": "human_agent",
    "home visit": "human_agent",
    "site visit": "human_agent",
    "engineer visit": "human_agent",
}


# Chroma / retrieval
CHROMA_COLLECTION_NAME: str = _env("CHROMA_COLLECTION_NAME", "mobilis_kb")
CHROMA_PERSIST_DIR: str = _env("CHROMA_PERSIST_DIR", "./chroma_persist")
RETRIEVAL_K: int = int(_env("RETRIEVAL_K", "5"))
OOD_DISTANCE_THRESHOLD: float = float(_env("OOD_DISTANCE_THRESHOLD", "0.8"))


# LLM backend configuration: set keys/models via env for your provider
LLM_BACKEND: str = _env("LLM_BACKEND", "groq")  # one of: openai, ollama, groq

# OpenAI
OPENAI_API_KEY: str | None = _env("OPENAI_API_KEY", _env("OPENAI_KEY", None))
OPENAI_MODEL: str = _env("OPENAI_MODEL", "gpt-4o-mini")

# Ollama
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "mistral")

# Groq (OpenAI-compatible endpoint)
GROQ_API_KEY: str | None = _env("GROQ_API_KEY", None)
GROQ_BASE_URL: str | None = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL: str | None = _env("GROQ_MODEL", "llama-3.3-70b-versatile")


if __name__ == "__main__":
    # Quick smoke-check when invoked directly
    print("Configuration preview:")
    print(f"EMBED_MODEL={EMBED_MODEL}")
    print(f"BITEXT_DATASET_PATH={BITEXT_DATASET_PATH}")
    print(f"INTENT_MODEL_PATH={INTENT_MODEL_PATH}")
    print(f"CHROMA_PERSIST_DIR={CHROMA_PERSIST_DIR}")
    print(f"CHROMA_COLLECTION_NAME={CHROMA_COLLECTION_NAME}")
    print(f"LLM_BACKEND={LLM_BACKEND}")
    print(f"GROQ_API_KEY set: {GROQ_API_KEY is not None}")
    print(f"GROQ_BASE_URL={GROQ_BASE_URL}")
    print(f"GROQ_MODEL={GROQ_MODEL}")