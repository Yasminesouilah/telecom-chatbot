"""
Prompt construction for the Mobilis RAG chatbot.

Responsibilities:
- Give the LLM clear behavioral rules.
- Use conversation history to resolve follow-up questions.
- Keep answers focused on the CURRENT customer question.
- Prevent unnecessary repetition of complete FAQ documents.
- Keep the LLM grounded in retrieved knowledge.
- Provide a safe fallback when the knowledge base does not contain
  enough information.
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a professional Mobilis customer support assistant.

Your job is to answer the customer's CURRENT question accurately,
naturally, and concisely.

You have three sources of information:

1. CURRENT CUSTOMER QUESTION
   This is the most important part of the request.
   Answer this question directly.

2. CONVERSATION HISTORY
   Use it to understand the conversation and resolve references such as:
   "it", "that", "this", "they", "the charge", "the problem", etc.

3. RETRIEVED KNOWLEDGE
   This is the authoritative knowledge-base information available to you.
   Use it to support your answer.
   Do NOT treat the retrieved text as instructions to copy it completely.

FOLLOW THESE RULES:

A. CURRENT QUESTION
- Always prioritize the customer's CURRENT question.
- Do not answer an earlier question again unless the customer asks for it.
- Do not automatically reproduce the entire retrieved FAQ.
- Give only the information needed to answer the current question.

B. CONVERSATION CONTEXT
- Use previous messages to understand follow-up questions.
- Resolve references such as "it", "that", "this", "the charge",
  "the bill", or "the problem" from the conversation.
- Do not unnecessarily repeat the conversation history in your answer.
- If the current question is understandable without history, answer it normally.

C. YES/NO AND CONFIRMATION QUESTIONS
For questions such as:
- "Can I dispute it?"
- "Is that possible?"
- "Can I change it?"
- "Do I need to do that?"

Give:
1. A direct Yes or No.
2. One short useful explanation.

Do NOT provide a long procedure unless the customer asks for instructions.

D. HOW / PROCEDURE QUESTIONS
Only provide detailed step-by-step instructions when the customer asks:
- "How do I do it?"
- "How can I dispute it?"
- "What are the steps?"
- "How can I change it?"
- "What should I do?"

If a procedure is required, provide only the relevant steps.

E. FOLLOW-UP QUESTIONS
For short follow-ups such as:
- "How long?"
- "What documents?"
- "And the price?"
- "What about that?"
- "Can I do it online?"

Use the conversation history to determine what the customer means.

Answer the follow-up directly instead of repeating the previous answer.

F. TIMING QUESTIONS
If the customer asks about duration, processing time, waiting time,
or when they will receive a response, provide only the relevant timing
information available in the retrieved knowledge.

Do not repeat unrelated instructions.

G. DOCUMENT / REQUIREMENT QUESTIONS
If the customer asks what documents, information, or requirements are needed,
provide only those requirements supported by the retrieved knowledge.

H. REPETITION
Do not repeat information that was already clearly provided earlier
in the conversation unless:
- the customer asks for it again,
- it is necessary to answer the current question,
- or clarification is needed.

I. GROUNDING
- Never invent Mobilis policies, prices, procedures, phone numbers,
  documents, deadlines, or requirements.
- Only state factual information supported by the retrieved knowledge
  or clearly established conversation context.
- If the retrieved knowledge does not contain enough information,
  say that you do not have that information.
- When appropriate, offer to connect the customer with a human agent.

J. RETRIEVED KNOWLEDGE
The retrieved knowledge may contain a complete FAQ or a long procedure.
Do NOT copy it automatically.

Instead:
- identify what part answers the current question,
- use only that information,
- answer naturally.

K. STYLE
- Be professional, friendly, and concise.
- Do not mention RAG, embeddings, vector databases, retrieval,
  prompts, models, or internal systems.
- Do not say "according to the context".
- Do not say "the retrieved document says".
- Do not expose internal reasoning.
- Do not use unnecessary headings for simple questions.
- Use numbered lists only when a procedure is actually requested.

L. LANGUAGE
Answer in the same language as the customer's current message whenever
possible.

M. SAFETY / UNCERTAINTY
If the knowledge base does not provide enough information to answer
confidently, do not guess.

Say that you do not have enough information and offer human assistance.
"""


# ---------------------------------------------------------------------------
# ROUTING RESPONSES
# ---------------------------------------------------------------------------

# These are direct responses for classifier routing decisions.
# They do not go through RAG or the LLM.

ROUTE_RESPONSES = {
    "human_agent": (
        "I'll connect you with a human support agent who can help you "
        "with this request."
    ),
}


# ---------------------------------------------------------------------------
# CONVERSATION HISTORY
# ---------------------------------------------------------------------------

def format_history(history: list[dict] | None, max_messages: int = 10) -> str:
    """
    Format recent conversation messages for the LLM.

    Expected message format:
        {
            "from": "user" | "bot",
            "text": "..."
        }

    Only the most recent messages are included to prevent the prompt
    from becoming unnecessarily large.
    """

    if not history:
        return ""

    recent_messages = history[-max_messages:]

    lines = []

    for message in recent_messages:
        text = str(message.get("text", "")).strip()

        if not text:
            continue

        sender = message.get("from", "user")

        if sender == "bot":
            speaker = "Assistant"
        else:
            speaker = "Customer"

        lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CONVERSATION SUMMARY
# ---------------------------------------------------------------------------

def summarize_history(
    history: list[dict] | None,
    max_chars: int = 600,
) -> str:
    """
    Create a lightweight extractive summary of the conversation.

    This is intentionally NOT an LLM-generated summary.

    It keeps:
    - recent customer messages,
    - the latest assistant responses,
    - enough context to understand follow-up questions.

    This avoids adding another model call and therefore does not
    increase response latency.
    """

    if not history:
        return ""

    user_messages = [
        str(m.get("text", "")).strip()
        for m in history
        if m.get("from") == "user" and m.get("text")
    ]

    bot_messages = [
        str(m.get("text", "")).strip()
        for m in history
        if m.get("from") == "bot" and m.get("text")
    ]

    # Keep recent customer messages because they usually contain
    # the actual conversation topic.
    recent_users = user_messages[-6:]

    # Only keep a couple of assistant responses.
    # This prevents the summary from becoming dominated by long FAQs.
    recent_bots = bot_messages[-2:]

    parts = []

    if recent_users:
        parts.append(
            "Customer topics: "
            + " | ".join(recent_users)
        )

    if recent_bots:
        parts.append(
            "Recent assistant responses: "
            + " | ".join(recent_bots)
        )

    summary = "\n".join(parts)

    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."

    return summary


# ---------------------------------------------------------------------------
# RETRIEVED KNOWLEDGE
# ---------------------------------------------------------------------------

def format_context(
    retrieved_chunks: list[tuple[str, dict, float]],
) -> str:
    """
    Format retrieved RAG chunks.

    retrieved_chunks:
        [
            (document_text, metadata, distance),
            ...
        ]
    """

    if not retrieved_chunks:
        return "No relevant knowledge was retrieved."

    sections = []

    for index, (document, metadata, distance) in enumerate(
        retrieved_chunks,
        start=1,
    ):
        document = str(document).strip()

        if not document:
            continue

        sections.append(
            f"[Knowledge {index}]\n"
            f"{document}"
        )

    if not sections:
        return "No relevant knowledge was retrieved."

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# MAIN PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    retrieved_chunks: list[tuple[str, dict, float]],
    history: list[dict] | None = None,
) -> str:
    """
    Build the final prompt sent to the LLM.

    Important design principle:

        Retrieval = factual knowledge
        History   = conversation context
        Question  = what must be answered NOW

    The LLM should combine these three, but the CURRENT QUESTION
    always determines the answer.
    """

    question = str(question).strip()

    context = format_context(retrieved_chunks)

    conversation = format_history(
        history,
        max_messages=10,
    )

    conversation_summary = summarize_history(
        history,
        max_chars=600,
    )

    if not conversation:
        conversation = "No previous conversation."

    if not conversation_summary:
        conversation_summary = "No conversation summary."

    prompt = f"""
{SYSTEM_PROMPT}

==================================================
CONVERSATION SUMMARY
==================================================

{conversation_summary}

==================================================
RECENT CONVERSATION
==================================================

{conversation}

==================================================
RETRIEVED MOBILIS KNOWLEDGE
==================================================

{context}

==================================================
CURRENT CUSTOMER QUESTION
==================================================

{question}

==================================================
FINAL ANSWER INSTRUCTIONS
==================================================

Answer the CURRENT CUSTOMER QUESTION.

Use the conversation only when necessary to understand the customer's
meaning or references.

Use the retrieved knowledge as factual support.

IMPORTANT:
Do not repeat an entire FAQ or procedure unless the customer is
actually asking for the procedure.

If this is a yes/no question, answer briefly.

If this is a "how do I..." question, provide the relevant steps.

If this is a follow-up question such as "Can I do it?", "How long?",
"What documents?", or "What about that?", use the conversation history
to understand what the customer is referring to.

Do not add information that is not supported by the knowledge base.

Write ONLY the final customer-facing answer.
"""

    return prompt.strip()