"""
Ties the whole RAG pipeline together.

setup() builds/loads the vector store once; chat() is the single
entry point everything else calls (CLI, future web API, etc.) —
mirrors answer_question() from 02_rag_pipeline_integrated.ipynb,
but with real generation wired in instead of returning the raw prompt.
"""

from config import CHROMA_COLLECTION_NAME, RETRIEVAL_K

from rag.chunking import build_chunks
from rag.classifier import predict_intent
from rag.data_loader import load_knowledge_base
from rag.embedding import get_embedding_fn
from rag.llm import generate_answer
from rag.prompts import build_prompt
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
    Full pipeline: classify intent -> filter retrieval -> build prompt
    -> generate answer.
    """
    global _collection
    if _collection is None:
        setup()

    intent_result = predict_intent(message)
    intent_filter = None if intent_result["intent"] == "unknown" else intent_result["intent"]

    retrieved = retrieve(_collection, message, k=k, intent_filter=intent_filter)
    prompt = build_prompt(message, retrieved)
    answer = generate_answer(prompt)

    if verbose:
        print(
            f"Predicted intent : {intent_result['intent']} "
            f"(raw={intent_result['raw_intent']}, confidence={intent_result['confidence']:.2f})"
        )
        print(f"Retrieval filter : {intent_filter}")
        print("-" * 70)

    return answer