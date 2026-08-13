"""
Loads the Bitext telco dataset as the RAG knowledge base.

This is the retrieval-side counterpart to 01_intent_classifier.ipynb's
dataset loading — same CSV, different purpose: here every row becomes
a retrievable chunk instead of a training example.
"""

import pandas as pd

from config import BITEXT_DATASET_PATH


def load_knowledge_base() -> pd.DataFrame:
    """
    Load and lightly clean the Bitext telco dataset.

    Returns a DataFrame with (at least) 'instruction', 'response',
    'intent', and 'category' columns — one row per Q/A pair.
    """
    df = pd.read_csv(BITEXT_DATASET_PATH)
    df = df.drop_duplicates().dropna()
    return df
