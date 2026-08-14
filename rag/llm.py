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


def _stream_openai(prompt: str):
    """Yield tokens from OpenAI-like streaming responses.

    Note: this assumes the OpenAI-compatible client supports streaming. If not,
    the wrapper will fall back to returning the full response as a single chunk.
    """
    # Do not swallow exceptions here — if streaming fails, surface the
    # error so callers can decide how to handle it. Attempt to extract
    # `content` robustly whether `delta` is an object with `.content` or
    # a mapping with a `get("content")` method.
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    # The exact streaming API varies by client; this is a best-effort
    # approximation for OpenAI-compatible streaming where `.stream=True`
    # yields events.
    for chunk in client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        text = None
        try:
            delta = chunk.choices[0].delta
            # Prefer attribute access
            text = getattr(delta, "content", None)
            if text is None:
                # Fallback to dict-like access
                try:
                    text = delta.get("content")
                except Exception:
                    text = None
        except Exception:
            text = None

        if text:
            yield text


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


def _stream_groq(prompt: str):
    """Stream from Groq's OpenAI-compatible endpoint using the OpenAI client
    configured with `base_url` and `api_key`.
    """
    from openai import OpenAI

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    for chunk in client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        text = None
        try:
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text is None:
                try:
                    text = delta.get("content")
                except Exception:
                    text = None
        except Exception:
            text = None

        if text:
            yield text


_BACKENDS = {"openai": _generate_openai, "ollama": _generate_ollama, "groq": _generate_groq}


def generate_answer(prompt: str, backend: str = LLM_BACKEND) -> str:
    """Generate the final answer using the configured LLM backend."""
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND '{backend}'. Use one of: {list(_BACKENDS)}")
    return _BACKENDS[backend](prompt)


def stream_answer(prompt: str, backend: str = LLM_BACKEND):
    """Stream the LLM answer if the backend supports it; otherwise yield the full answer once."""
    if backend == "openai":
        yield from _stream_openai(prompt)
    elif backend == "groq":
        yield from _stream_groq(prompt)
    else:
        # Other backends currently don't support true streaming in this codebase.
        # Fall back to the synchronous generator that yields the full response.
        full = generate_answer(prompt, backend=backend)
        yield full