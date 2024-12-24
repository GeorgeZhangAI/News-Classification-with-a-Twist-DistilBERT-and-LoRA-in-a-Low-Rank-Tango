import os
import torch
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
from data_utils import (
    load_data,
    tokenize_data,
    get_tokenizer,
    create_label_mapping,
    process_labels
)
from train_utils import train_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataset(Dataset):
    """Custom dataset for handling tokenized data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def get_lora_model(model_name, num_labels):
    """Returns a model with LoRA adaptation for specific modules."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=["q_lin", "v_lin", "ffn.lin1", "ffn.lin2"]
    )

    model = get_peft_model(model, lora_config)
    return model


def main():
    # File paths
    train_csv = "../data/train.csv"
    test_csv = "../data/test.csv"
    model_name = "distilbert-base-uncased"

    # Device 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Load and tokenize data
    tokenizer = get_tokenizer(model_name)
    train_texts, train_labels = load_data(train_csv)
    test_texts, test_labels = load_data(test_csv)

    
    # Create label mapping and process labels
    label_mapping = create_label_mapping(train_labels)
    print(f"Label Mapping: {label_mapping}")
    
    train_labels = process_labels(train_labels, label_mapping)
    test_labels = process_labels(test_labels, label_mapping)

    
    # Tokenize data
    train_encodings = tokenize_data(train_texts, tokenizer)
    test_encodings = tokenize_data(test_texts, tokenizer)

    
    # Prepare datasets and loaders
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)


    
    # Initialize model with LoRA
    num_labels = len(label_mapping)
    
    model = get_lora_model(model_name, num_labels).to(device)
            
    # Train and evaluate
    train_model(model, train_loader, test_loader, device, epochs=3)

if __name__ == "__main__":
    main()