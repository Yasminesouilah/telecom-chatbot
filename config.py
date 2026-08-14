"""
Central configuration for the Mobilis Telecom RAG chatbot.

Every tunable knob lives here so the rest of the codebase never
hardcodes a model name, threshold, or path. Change behavior by
editing this file (or overriding with environment variables) —
never by editing rag/*.py directly.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in this folder into os.environ, if present

# ── Data ──────────────────────────────────────────────────────────
BITEXT_DATASET_PATH = (
    "hf://datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/"
    "bitext-telco-llm-chatbot-training-dataset.csv"
)

# ── Intent classifier ────────────────────────────────────────────
# Path to the model saved by 01_intent_classifier.ipynb (section 13).
INTENT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobilis_intent_model")

# Below this confidence, predict_intent() returns "unknown" instead
# of a shaky label. Keep this in sync across classifier + RAG code —
# it's why this lives here instead of being duplicated in both places.
CONFIDENCE_THRESHOLD = 0.5

# ── Embeddings ────────────────────────────────────────────────────
# Multilingual by default (Mobilis serves Algerian customers — French/
# Arabic input is likely). Swap to "all-MiniLM-L6-v2" for English-only,
# slightly faster embeddings.
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Vector store ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "mobilis_kb"
RETRIEVAL_K = 3  # how many chunks to retrieve per query

# ── LLM generation ────────────────────────────────────────────────
# "openai", "ollama", or "groq" — generate_answer() in rag/llm.py branches on this.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # never hardcode the key itself

OLLAMA_MODEL = "mistral"

# Groq: free, no-credit-card tier. OpenAI-compatible API, so it reuses
# the same client code as the OpenAI backend with a different base_url.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # never hardcode the key itself
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MARGIN_THRESHOLD = 0.15   # min gap between top-1 and top-2 score
OUT_OF_SCOPE_KEYWORDS = {
    "technician": "human_agent",
    "technician visit": "human_agent",
    "home visit": "human_agent",
    "site visit": "human_agent",
    "engineer visit": "human_agent",
}
OOD_DISTANCE_THRESHOLD = 1.0  # placeholder — needs empirical tuning, see below