"""
Sets up the embedding function used to vectorize chunks and queries.

Ported from 02_rag_pipeline_integrated.ipynb section 5. Defaults to
the multilingual MiniLM model (config.EMBED_MODEL) since Mobilis
customers may write in French or Arabic as well as English.

NOTE on the (removed) fallback: this used to fall back to ChromaDB's
built-in embedding function if sentence-transformers failed to load.
That's dangerous for a persistent vector store: if the database was
built with MiniLM embeddings and a later run silently falls back to
Chroma's default embedder (e.g. after a dependency break on some
machine), every query vector lands in a different embedding space
than the stored chunk vectors. Distances become meaningless and
retrieval degrades silently — no error, just bad answers. There must
be exactly ONE embedding model across build time and query time, so
failure to load it now raises instead of substituting a different
model.
"""

from chromadb.utils import embedding_functions

from config import EMBED_MODEL


def get_embedding_fn():
    """
    Build (and sanity-check) the embedding function.

    Forces a real embedding call now, at setup time, so any
    version-mismatch or download error surfaces here — as a hard
    failure — rather than silently switching embedding models later
    during retrieval.
    """
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        embedding_fn(["test"])  # force model load now, not on first real query
        print(f"Using sentence-transformers: {EMBED_MODEL}")
        return embedding_fn
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{EMBED_MODEL}' "
            f"({type(e).__name__}: {e}). Refusing to silently fall back to a "
            "different embedding model, since that would desync the vector "
            "space from whatever built the existing ChromaDB collection. "
            "Fix the sentence-transformers install/model download and retry."
        ) from e