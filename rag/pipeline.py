"""
Ties the whole RAG pipeline together.

setup() builds/loads the vector store once; chat() is the single
entry point everything else calls (CLI, future web API, etc.) —
mirrors answer_question() from 02_rag_pipeline_integrated.ipynb,
but with real generation wired in instead of returning the raw prompt.
"""

from config import CHROMA_COLLECTION_NAME, OOD_DISTANCE_THRESHOLD, RETRIEVAL_K

from rag.chunking import build_chunks
from rag.classifier import predict_intent
from rag.data_loader import load_knowledge_base
from rag.embedding import get_embedding_fn
from rag.llm import generate_answer, stream_answer
from rag.prompts import ROUTE_RESPONSES, build_prompt
from rag.vectorstore import build_or_load_collection, get_client, retrieve

_collection = None  # lazy-loaded singleton


def setup(rebuild: bool = False):
    """
    Build (or load) the vector store collection. Call this once at
    startup — run_chat.py does this automatically on first use.

    rebuild=True forces re-embedding the knowledge base from scratch
    (needed if the underlying dataset or embedding model changes).
    On a fresh run (no collection persisted yet), chunks are built
    automatically even without rebuild=True — the collection existing
    on disk is what actually matters, not the flag alone.
    """
    global _collection

    client = get_client()
    embedding_fn = get_embedding_fn()

    existing = [c.name for c in client.list_collections()]
    need_chunks = rebuild or CHROMA_COLLECTION_NAME not in existing

    chunks = metadatas = ids = None
    if need_chunks:
        df = load_knowledge_base()
        chunks, metadatas, ids = build_chunks(df)

    _collection = build_or_load_collection(
        client, embedding_fn, chunks=chunks, metadatas=metadatas, ids=ids, rebuild=rebuild
    )
    return _collection


def chat(message: str, k: int = RETRIEVAL_K, verbose: bool = True, history: list[dict] | None = None) -> str:
    """
    Full pipeline: classify intent -> OOD distance gate -> filter retrieval
    -> build prompt -> generate answer.

    The classifier's own confidence/margin turned out not to be a reliable
    OOD signal (it can hit 0.99 on messages with no matching intent at all —
    e.g. "sim card stopped working" scored 0.99 on payment_methods). So this
    adds an independent check: retrieve unfiltered first and look at how
    close the *nearest* chunk in the whole knowledge base actually is. If
    nothing is close, the message is genuinely out of scope regardless of
    what the classifier claims, and the classifier's intent is overridden.

    OOD_DISTANCE_THRESHOLD needs empirical tuning for your embedding model/
    metric — print top_distance (see verbose output below) for a batch of
    known in-domain and known out-of-domain messages and pick a cutoff
    between the two clusters.

    `route` (e.g. "human_agent") is a routing decision, not a knowledge-base
    intent — it never exists as chunk metadata in Chroma, so it must be
    handled BEFORE any retrieval happens, not passed in as an intent_filter
    (that would return zero chunks and silently break the response).
    """
    # Use the centralized prepare_prompt to avoid duplication.
    prompt_or_answer, retrieved, route_answer = prepare_prompt(message, k=k, history=history)

    if route_answer is not None:
        if verbose:
            print(f"Route            : {route_answer} (no retrieval, no LLM call)")
            print("-" * 70)
        return route_answer

    # prompt_or_answer is the assembled prompt string here
    prompt = prompt_or_answer
    answer = generate_answer(prompt)

    if verbose:
        # Try to print some diagnostic info if available from retrieved
        try:
            top_distance = retrieved[0][2] if retrieved else float("inf")
        except Exception:
            top_distance = float("inf")
        intent_filter = None
        print("-" * 70)
        print(f"Top retrieval dist: {top_distance:.4f} (OOD threshold={OOD_DISTANCE_THRESHOLD})")
        print(f"Retrieval filter : {intent_filter}")
        print("-" * 70)

    return answer


def prepare_prompt(message: str, k: int = RETRIEVAL_K, history: list[dict] | None = None):
    """Run classifier/retrieval and return the assembled prompt and metadata.

    Returns a 3-tuple: (prompt_or_none, retrieved_list_or_none, route_answer_or_none).
    - If a route is detected, `route_answer_or_none` will be a string and the
      other two values will be None or empty. Callers should short-circuit
      and return/stream the canned `route_answer`.
    - Otherwise `prompt_or_none` is the assembled prompt and `retrieved_list_or_none`
      is the retrieval metadata list.
    """
    global _collection
    if _collection is None:
        setup()

    context_for_intent = message
    if history:
        for m in reversed(history):
            if m.get("from") == "user" and m.get("text"):
                context_for_intent = m.get("text") + "\n" + message
                break

    intent_result = predict_intent(context_for_intent)

    # Handle explicit routing decisions early and return a canned answer
    # so callers (sync or streaming) don't invoke the LLM.
    if intent_result["route"] is not None:
        route_answer = ROUTE_RESPONSES.get(intent_result["route"], ROUTE_RESPONSES["human_agent"])
        return None, None, route_answer

    intent_filter = None if intent_result["intent"] == "unknown" else intent_result["intent"]

    retrieval_query = message

    broad_retrieved = retrieve(_collection, retrieval_query, k=k, intent_filter=None)
    top_distance = broad_retrieved[0][2] if broad_retrieved else float("inf")

    if top_distance > OOD_DISTANCE_THRESHOLD:
        intent_filter = None
        retrieved = broad_retrieved
    elif intent_filter is not None:
        retrieved = retrieve(_collection, message, k=k, intent_filter=intent_filter)
    else:
        retrieved = broad_retrieved

    prompt = build_prompt(message, retrieved, history=history)
    return prompt, retrieved, None


def stream_chat(message: str, k: int = RETRIEVAL_K, history: list[dict] | None = None, backend: str | None = None):
    """Stream tokens for a chat request by reusing prepare_prompt and
    `rag.llm.stream_answer`. Yields strings (tokens/chunks) as produced
    by the backend.
    """
    prompt, retrieved, route_answer = prepare_prompt(message, k=k, history=history)

    # If a route was detected, stream the canned response and return.
    if route_answer is not None:
        # Stream the route answer in one piece to keep client simple.
        yield route_answer
        return

    # Use configured backend unless caller overrides.
    if backend is None:
        from config import LLM_BACKEND as _cfg_backend

        backend = _cfg_backend

    for token in stream_answer(prompt, backend=backend):
        yield token