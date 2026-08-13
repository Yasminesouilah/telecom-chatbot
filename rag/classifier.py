"""
Loads the trained intent classifier and predicts intent with a
confidence fallback.

Ported from 01_intent_classifier.ipynb section 14 / the integration
cell in 02_rag_pipeline_integrated.ipynb. Requires mobilis_intent_model/
to exist at config.INTENT_MODEL_PATH (produced by 01_intent_classifier.ipynb).
"""

from transformers import pipeline

from config import CONFIDENCE_THRESHOLD, INTENT_MODEL_PATH

_classifier = None  # lazy-loaded singleton — model load is expensive


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=INTENT_MODEL_PATH,
            tokenizer=INTENT_MODEL_PATH,
        )
    return _classifier


def predict_intent(message: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Classify a customer message's intent.

    If the model's top prediction is below `threshold`, returns
    intent="unknown" so downstream retrieval falls back to an
    unfiltered (broad) search instead of trusting a shaky label.

    Returns: {"intent": str, "raw_intent": str, "confidence": float}
    """
    classifier = _get_classifier()
    prediction = classifier(message.lower())[0]
    intent = prediction["label"]
    score = prediction["score"]

    if score < threshold:
        return {"intent": "unknown", "raw_intent": intent, "confidence": score}
    return {"intent": intent, "raw_intent": intent, "confidence": score}
