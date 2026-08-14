"""
System prompt and prompt assembly for the LLM generation step.

Ported from 02_rag_pipeline_integrated.ipynb section 10. The "ONLY
the context below" instruction plus an explicit escalate-to-human
fallback is what keeps the LLM from hallucinating outside the
Bitext-derived knowledge base.
"""

SYSTEM_PROMPT = (
    "You are a Mobilis customer support assistant. Answer the customer's "
    "question using ONLY the context below. If the context does not "
    "contain the answer, say you don't have that information and offer "
    "to escalate to a human agent."
)

# Canned responses for routing decisions (classifier.py's `route` field).
# These are NOT knowledge-base intents — no chunk is ever tagged with
# them — so they're handled with a direct response instead of retrieval
# + LLM generation. Add an entry here for every route value used in
# config.OUT_OF_SCOPE_KEYWORDS.
ROUTE_RESPONSES = {
    "human_agent": (
        "That's something a human agent needs to help with directly (like "
        "scheduling a technician or site visit). I'll connect you with our "
        "support team, who can arrange that for you."
    ),
}


def build_prompt(question: str, retrieved_chunks: list[tuple[str, dict, float]]) -> str:
    """
    Assemble the final prompt sent to the LLM.

    retrieved_chunks is the output of vectorstore.retrieve():
    a list of (document_text, metadata, distance) tuples.
    """
    context = "\n\n".join(doc for doc, _meta, _dist in retrieved_chunks)

    return f"""{SYSTEM_PROMPT}

Context:
{context}

Customer question: {question}

Answer in a natural, helpful, concise tone:"""