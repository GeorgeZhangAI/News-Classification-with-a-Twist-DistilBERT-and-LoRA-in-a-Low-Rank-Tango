# README

## Project Structure

```plaintext
project_root/
├── bert (advanced)/         # Primary model: LoRA-enhanced DistilBERT
│   ├── data_utils.py        # Data preprocessing: tokenization, label mapping.
│   ├── lora_finetune.py     # Main script: fine-tuning and evaluation.
│   └── train_utils.py       # Training and evaluation functions.
├── data/                    # Dataset files
│   ├── test.csv             # Test set.
│   └── train.csv            # Training set.
└── fasttext (baseline)/     # Baseline model: FastText
    ├── main.py              # Main script: training and evaluation.
    └── preprocessing.py     # Data preprocessing for FastText.
```

## How to Run
> Note: Make sure to place all files in the directory structure exactly as shown above before running the scripts.

### Baseline Model: FastText
1. Navigate to the FastText directory: `cd fasttext\ \(baseline\)`
2. Run the training and evaluation script: `python main.py`

### Primary Model: LoRA-Enhanced DistilBERT
1. Navigate to the BERT directory: `cd bert\ \(advanced\)`
2. Execute the fine-tuning and evaluation script: `python lora_finetune.py`



### Notes
- Ensure all dependencies are installed as specified in the requirements.txt file (if provided).
- Use Python 3.7 or higher for compatibility.
- Adjust file paths in the scripts if using custom datasets or modified directory structures.
