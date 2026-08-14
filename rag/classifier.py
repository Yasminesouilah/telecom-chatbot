"""
Loads the trained intent classifier and predicts intent with a
confidence fallback.

Ported from 01_intent_classifier.ipynb section 14 / the integration
cell in 02_rag_pipeline_integrated.ipynb. Requires mobilis_intent_model/
to exist at config.INTENT_MODEL_PATH (produced by 01_intent_classifier.ipynb).

NOTE on the confidence fallback: the training set (Bitext) is heavily
templated (see the near-duplicate check in 01_intent_classifier.ipynb),
so the model is overconfident by construction — top-1 softmax scores
sit near 1.00 even on inputs that don't match any of the 26 trained
intents (e.g. "schedule a technician visit" has no matching intent in
the dataset, but "schedule" only co-occurs with schedule_payments, so
the model confidently mislabels it). A raw top-1 threshold can't catch
this. Two extra safeguards are added below:

1. Margin check: compare top-1 vs. top-2 score instead of just top-1.
   A genuinely OOD input often still has meaningful mass on other
   classes even when top-1 looks saturated.
2. Keyword pre-filter: known out-of-scope topics (not covered by any
   of the 26 intents at all, e.g. technician dispatch) are caught
   before the classifier runs at all, since no amount of confidence
   calibration fixes a label that doesn't exist in the training data.

NOTE on "intent" vs. "route": the 26 trained intents (cancel_plan,
dispute_invoice, ...) are the only labels that exist as metadata in
the ChromaDB knowledge base. Something like "human_agent" is NOT a
knowledge-base intent — it's a routing decision (escalate, don't
search). Treating it as an intent and handing it to Chroma's
`where={"intent": "human_agent"}` filter returns zero chunks, because
no chunk was ever tagged that way. So `route` is returned as a
separate field from `intent`: pipeline.py checks `route` first and,
if set, skips retrieval/filtering entirely and goes straight to an
escalation response.

Add these to config.py:
    MARGIN_THRESHOLD = 0.15   # min gap between top-1 and top-2 score
    OUT_OF_SCOPE_KEYWORDS = {
        "technician": "human_agent",
        "technician visit": "human_agent",
        "home visit": "human_agent",
        "site visit": "human_agent",
        "engineer visit": "human_agent",
    }
"""

from transformers import pipeline

from config import (
    CONFIDENCE_THRESHOLD,
    INTENT_MODEL_PATH,
    MARGIN_THRESHOLD,
    OUT_OF_SCOPE_KEYWORDS,
)

_classifier = None  # lazy-loaded singleton — model load is expensive


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=INTENT_MODEL_PATH,
            tokenizer=INTENT_MODEL_PATH,
            top_k=None,  # return scores for every label, not just the top one
        )
    return _classifier


def _check_out_of_scope(message: str) -> str | None:
    """
    Catch topics that aren't covered by ANY trained intent at all.
    No confidence threshold can fix a missing label — this has to be
    a hard pre-filter, checked before the classifier runs.

    Returns a route (e.g. "human_agent") if a known out-of-scope
    keyword is found, else None.
    """
    lowered = message.lower()
    for keyword, route in OUT_OF_SCOPE_KEYWORDS.items():
        if keyword in lowered:
            return route
    return None


def predict_intent(
    message: str,
    threshold: float = CONFIDENCE_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> dict:
    """
    Classify a customer message's intent.

    `route` (separate from `intent`) is set when the message matches a
    known out-of-scope keyword. It is never one of the 26 trained/
    knowledge-base intents, so callers must check `route` BEFORE using
    `intent` to filter retrieval — a route means "don't search the KB,
    handle this some other way" (e.g. escalate to a human agent).

    `intent` falls back to "unknown" (broad, unfiltered retrieval) when:
    - a known out-of-scope keyword pre-filter matches (route is set), OR
    - the top prediction is below `threshold`, OR
    - the gap between the top-1 and top-2 scores is below
      `margin_threshold` — i.e. the model is torn between two labels
      even if the top score alone looks saturated.

    Returns: {"intent": str, "raw_intent": str | None, "confidence": float, "route": str | None}
    """
    route = _check_out_of_scope(message)
    if route is not None:
        return {"intent": "unknown", "raw_intent": None, "confidence": 0.0, "route": route}

    classifier = _get_classifier()
    predictions = sorted(classifier(message.lower())[0], key=lambda p: p["score"], reverse=True)
    top, second = predictions[0], predictions[1]
    intent = top["label"]
    score = top["score"]
    margin = score - second["score"]

    if score < threshold or margin < margin_threshold:
        return {"intent": "unknown", "raw_intent": intent, "confidence": score, "route": None}
    return {"intent": intent, "raw_intent": intent, "confidence": score, "route": None}