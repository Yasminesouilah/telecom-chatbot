"""
Sets up the embedding function used to vectorize chunks and queries.

Ported from 02_rag_pipeline_integrated.ipynb section 5. Defaults to
the multilingual MiniLM model (config.EMBED_MODEL) since Mobilis
customers may write in French or Arabic as well as English. Falls
back to ChromaDB's built-in embedding function if sentence-transformers
fails to load, so a version mismatch surfaces as a printed warning
instead of a hard crash.
"""

from chromadb.utils import embedding_functions

from config import EMBED_MODEL


def get_embedding_fn():
    """
    Build (and sanity-check) the embedding function.

    Forces a real embedding call now, at setup time, so any
    version-mismatch or download error surfaces here rather than
    silently later during retrieval.
    """
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        embedding_fn(["test"])  # force model load now, not on first real query
        print(f"Using sentence-transformers: {EMBED_MODEL}")
        return embedding_fn
    except Exception as e:
        print(
            f"sentence-transformers embedding failed ({type(e).__name__}: {e}). "
            "Falling back to ChromaDB's built-in embedding model instead."
        )
        return embedding_functions.DefaultEmbeddingFunction()
