import os
import torch
from tqdm import tqdm
from torch.optim import AdamW 
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def train_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    """Trains the model."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Training loss: {total_loss / len(train_loader)}")
        evaluate_model(model, val_loader, device)

    
    model_save_path = "./saved_model"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))
    model.config.to_json_file(os.path.join(model_save_path, "config.json"))
    
    print("Model saved at:", model_save_path)

        
def evaluate_model(model, data_loader, device):
    """Evaluates the model."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, axis=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Accuracy, Precision, Recall, F1 Score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
