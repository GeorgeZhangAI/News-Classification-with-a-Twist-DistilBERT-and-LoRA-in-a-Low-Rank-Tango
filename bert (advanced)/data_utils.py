import os
import pandas as pd
from transformers import AutoTokenizer


def load_data(csv_path):
    """Loads data from a CSV file."""
    df = pd.read_csv(csv_path).dropna(subset=['Description', 'Class Index'])
    texts = df['Description'].tolist()
    labels = df['Class Index'].tolist()
    return texts, labels


def create_label_mapping(labels):
    """Creates a mapping from original labels to 0-based integer indices."""
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_idx


def process_labels(labels, label_mapping):
    """Maps original labels to 0-based indices."""
    return [label_mapping[label] for label in labels]


def get_tokenizer(model_name="distilbert-base-uncased"):
    """Returns a tokenizer for the specified model."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_data(texts, tokenizer, max_length=128):
    """Tokenizes the data using a tokenizer."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
