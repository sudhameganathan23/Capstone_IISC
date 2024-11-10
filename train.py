import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_loader import download_log_data, preprocess_logs, create_sequences
from model import LogBERT, LogBERTTrainer
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogDataset(Dataset):
    def __init__(self, sequences: List[List[int]], labels: List[int], templates: dict, tokenizer: BertTokenizer):
        self.sequences = sequences
        self.labels = labels
        self.templates = templates
        self.tokenizer = tokenizer

        # Print first sequence tokenization for debugging
        if len(sequences) > 0:
            print("\nTokenization example for first sequence:")
            template_texts = []
            for template_id in sequences[0]:
                template = self.templates['EventTemplate'].iloc[template_id]
                template_texts.append(template)
            sequence_text = " [SEP] ".join(template_texts)
            print(f"Original text: {sequence_text[:200]}...")
            tokens = tokenizer(
                sequence_text,
                padding='max_length',
                max_length=128,  # Reduced from 512
                truncation=True,
                return_tensors='pt'
            )
            print(f"Tokenized form (first 50 tokens): {tokens['input_ids'][0][:50]}")
            decoded = tokenizer.decode(tokens['input_ids'][0][:50])
            print(f"Decoded back: {decoded}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert template IDs to actual template text
        template_texts = []
        for template_id in sequence:
            template = self.templates['EventTemplate'].iloc[template_id]
            template_texts.append(template)
        
        # Join templates with separator
        sequence_text = " [SEP] ".join(template_texts)
        
        # Tokenize sequence
        tokens = self.tokenizer(
            sequence_text,
            padding='max_length',
            max_length=128,  # Reduced from 512
            truncation=True,
            return_tensors='pt'
        )
        
        return (
            tokens['input_ids'].squeeze(),
            tokens['attention_mask'].squeeze(),
            torch.tensor(label)
        )

def plot_metrics(train_losses: List[float], val_losses: List[float], val_f1s: List[float]):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Download and preprocess data
    logger.info("Downloading and preprocessing data...")
    structured_logs, templates, _ = download_log_data()
    merged_logs, templates = preprocess_logs(structured_logs, templates)
    
    # Create sequences
    sequences, labels = create_sequences(merged_logs)
    
    # Take a smaller subset for faster training
    subset_size = min(1000, len(sequences))  # Use at most 1000 sequences
    sequences = sequences[:subset_size]
    labels = labels[:subset_size]
    
    # Split data
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = LogDataset(train_sequences, train_labels, templates, tokenizer)
    val_dataset = LogDataset(val_sequences, val_labels, templates, tokenizer)
    
    # Create dataloaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced from 32
    val_loader = DataLoader(val_dataset, batch_size=16)  # Reduced from 32
    
    # Initialize model
    model = LogBERT(vocab_size=tokenizer.vocab_size)
    trainer = LogBERTTrainer(model, device)
    
    # Training loop with fewer epochs
    num_epochs = 3  # Reduced from 10
    train_losses = []
    val_losses = []
    val_f1s = []
    best_f1 = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            loss = trainer.train_step(batch)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info("Saved new best model")
        
        # Plot metrics
        plot_metrics(train_losses, val_losses, val_f1s)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
