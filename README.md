# Project Overview

This project contains implementations of two models for text classification:
1. **Advanced Model**: LoRA-enhanced DistilBERT.
2. **Baseline Model**: FastText.

Both models are trained and evaluated on the same dataset for performance comparison.

## Directory Structure

```plaintext
project_root/
├── bert (advanced)/         # Primary model: LoRA-enhanced DistilBERT
│   ├── data_utils.py        # Handles data preprocessing such as tokenization and label mapping.
│   ├── lora_finetune.py     # Main script for fine-tuning DistilBERT with LoRA and evaluating its performance.
│   └── train_utils.py       # Contains helper functions for training and evaluation (e.g., loss computation, metrics).
├── data/                    # Directory containing dataset files.
│   ├── test.csv             # Test set.
│   └── train.csv            # Training set.
└── fasttext (baseline)/     # Baseline model using FastText.
    ├── main.py              # Main script for training and evaluating the FastText model.
    └── preprocessing.py     # Preprocesses the data for FastText (e.g., cleaning, tokenization).
