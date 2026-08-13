"""
ChromaDB wrapper: client/collection setup plus add() and retrieve().

Unlike the standalone notebook (which used an in-memory chromadb.Client()
that vanished each session), this uses a PersistentClient so the
knowledge base only needs to be built once — see build_or_load_collection().
"""

import chromadb

from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, RETRIEVAL_K


def get_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def build_or_load_collection(client, embedding_fn, chunks=None, metadatas=None, ids=None,
                              rebuild=False, batch_size=200):
    """
    Get the mobilis_kb collection, creating and populating it if needed.

    - If the collection already exists and rebuild=False, reuse it as-is
      (fast path — no re-embedding on every run).
    - If rebuild=True, or the collection doesn't exist yet, (re)create it
      and insert chunks/metadatas/ids in batches.
    """
    existing = [c.name for c in client.list_collections()]

    if CHROMA_COLLECTION_NAME in existing and not rebuild:
        print(f"Loading existing collection '{CHROMA_COLLECTION_NAME}'")
        return client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_fn)

    if CHROMA_COLLECTION_NAME in existing:
        client.delete_collection(CHROMA_COLLECTION_NAME)

    if chunks is None:
        raise ValueError(
            "No existing collection found and no chunks provided to build one. "
            "Pass chunks/metadatas/ids (see rag.chunking.build_chunks)."
        )

    collection = client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_fn)

    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    print(f"Stored {collection.count()} chunks in '{CHROMA_COLLECTION_NAME}'")
    return collection


def retrieve(collection, query: str, k: int = RETRIEVAL_K, intent_filter: str | None = None):
    """
    Retrieve the top-k chunks for a query, optionally filtered by intent.

    intent_filter=None searches the whole knowledge base (use this when
    the classifier returned "unknown" / low confidence). Pass an intent
    string to restrict the search once the classifier's prediction is trusted.

    Returns a list of (document_text, metadata, distance) tuples.
    """
    where = {"intent": intent_filter} if intent_filter else None
    results = collection.query(query_texts=[query], n_results=k, where=where)
    return list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))
