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
from rag.llm import generate_answer
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


def chat(message: str, k: int = RETRIEVAL_K, verbose: bool = True) -> str:
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
    global _collection
    if _collection is None:
        setup()

    intent_result = predict_intent(message)

    if intent_result["route"] is not None:
        answer = ROUTE_RESPONSES.get(
            intent_result["route"], ROUTE_RESPONSES["human_agent"]
        )
        if verbose:
            print(f"Route            : {intent_result['route']} (no retrieval, no LLM call)")
            print("-" * 70)
        return answer

    intent_filter = None if intent_result["intent"] == "unknown" else intent_result["intent"]

    # Always retrieve unfiltered first — this is the OOD signal, independent
    # of whatever the classifier believes.
    broad_retrieved = retrieve(_collection, message, k=k, intent_filter=None)
    top_distance = broad_retrieved[0][2] if broad_retrieved else float("inf")

    if top_distance > OOD_DISTANCE_THRESHOLD:
        # Nothing in the KB is actually close to this message — don't trust
        # the classifier's label even if it reported high confidence.
        intent_result["intent"] = "unknown"
        intent_filter = None
        retrieved = broad_retrieved
    elif intent_filter is not None:
        retrieved = retrieve(_collection, message, k=k, intent_filter=intent_filter)
    else:
        retrieved = broad_retrieved

    prompt = build_prompt(message, retrieved)
    answer = generate_answer(prompt)

    if verbose:
        print(
            f"Predicted intent : {intent_result['intent']} "
            f"(raw={intent_result['raw_intent']}, confidence={intent_result['confidence']:.2f})"
        )
        print(f"Top retrieval dist: {top_distance:.4f} (OOD threshold={OOD_DISTANCE_THRESHOLD})")
        print(f"Retrieval filter : {intent_filter}")
        print("-" * 70)

    return answer