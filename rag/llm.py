"""
Generation backend: turns an assembled prompt into a final answer.

02_rag_pipeline_integrated.ipynb left this as a commented-out stub
with two options (OpenAI vs. Ollama). This module implements both,
switchable via config.LLM_BACKEND, so pipeline.py never needs to
know which one is active.
"""

from config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_BACKEND,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def _generate_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _generate_ollama(prompt: str) -> str:
    import ollama

    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def _generate_groq(prompt: str) -> str:
    # Groq's API is OpenAI-compatible, so this reuses the same `openai`
    # client library — just pointed at Groq's base_url with a Groq key.
    from openai import OpenAI

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


_BACKENDS = {"openai": _generate_openai, "ollama": _generate_ollama, "groq": _generate_groq}


def generate_answer(prompt: str, backend: str = LLM_BACKEND) -> str:
    """Generate the final answer using the configured LLM backend."""
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND '{backend}'. Use one of: {list(_BACKENDS)}")
    return _BACKENDS[backend](prompt)