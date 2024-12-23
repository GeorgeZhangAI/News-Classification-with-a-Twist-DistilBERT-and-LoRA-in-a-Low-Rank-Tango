# README

## Project Structure

project_root/
├── bert (advanced)/  # Primary model: LoRA-enhanced DistilBERT
│   ├── data_utils.py    # Data preprocessing: tokenization, label mapping.
│   ├── lora_finetune.py # Main script: fine-tuning and evaluation.
│   └── train_utils.py   # Training and evaluation functions.
├── data/              # Dataset files
│   ├── test.csv         # Test set.
│   └── train.csv        # Training set.
└── fasttext (baseline)/ # Baseline model: FastText
    ├── main.py          # Main script: training and evaluation.
    └── preprocessing.py # Data preprocessing for FastText.



## How to Run

### Note: Make sure to place all files in the directory structure exactly as shown above before running the scripts.

### Baseline Model: FastText
1. Navigate to `fasttext (baseline)`: cd fasttext\ \(baseline\)
2. Run the script: python main.py

### Primary Model: LoRA-Enhanced DistilBERT
1. Navigate to bert (advanced): cd bert\ \(advanced\)
2. Run the script: python lora_finetune.py