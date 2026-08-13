"""
Turns knowledge-base rows into retrievable text chunks.

Ported directly from 02_rag_pipeline_integrated.ipynb section 4.
One Bitext row = one chunk (responses are short; no token-window
splitting needed). If real Mobilis documents get added later
(PDFs, policy pages), those will need a proper chunker — see the
note in the original notebook.
"""

import uuid

import pandas as pd


def build_chunk(row: pd.Series) -> str:
    """One chunk = the answer text, with light context prepended
    for retrieval quality (embedding captures topic, not just
    surface wording)."""
    return (
        f"Category: {row['category']} | Intent: {row['intent']}\n"
        f"Q: {row['instruction']}\n"
        f"A: {row['response']}"
    )


def build_chunks(df: pd.DataFrame) -> tuple[list[str], list[dict], list[str]]:
    """
    Build chunks + metadata + ids for every row in df.

    Returns three parallel lists ready to hand to
    vectorstore.add_chunks(): (chunks, metadatas, ids).
    """
    chunks, metadatas, ids = [], [], []

    for _, row in df.iterrows():
        chunks.append(build_chunk(row))
        metadatas.append({"intent": row["intent"], "category": row["category"]})
        ids.append(str(uuid.uuid4()))

    return chunks, metadatas, ids
