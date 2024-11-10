import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Tuple

class LogBERT(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768, num_layers: int = 12):
        super(LogBERT, self).__init__()
        
        # BERT configuration
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        
        # BERT model for log sequence modeling
        self.bert = BertModel(self.config)
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens to attend to
        Returns:
            Tuple of anomaly scores and sequence embeddings
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Compute anomaly score
        anomaly_score = self.anomaly_detector(pooled_output)
        
        return anomaly_score, sequence_output

class LogBERTTrainer:
    def __init__(self, model: LogBERT, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """
        Perform one training step
        Args:
            batch: Tuple of (input_ids, attention_mask, labels)
        Returns:
            Training loss
        """
        input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
        
        self.optimizer.zero_grad()
        anomaly_scores, _ = self.model(input_ids, attention_mask)
        loss = self.criterion(anomaly_scores.squeeze(), labels.float())
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens to attend to
        Returns:
            Tensor of anomaly scores
        """
        self.model.eval()
        with torch.no_grad():
            anomaly_scores, _ = self.model(input_ids.to(self.device), 
                                         attention_mask.to(self.device))
        return anomaly_scores.cpu()

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """
        Evaluate the model on a dataset
        Args:
            dataloader: DataLoader containing evaluation data
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                anomaly_scores, _ = self.model(input_ids, attention_mask)
                loss = self.criterion(anomaly_scores.squeeze(), labels.float())
                
                total_loss += loss.item()
                predictions.extend((anomaly_scores.squeeze() > 0.5).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
        
        # Calculate F1 score
        tp = sum(p == t == 1 for p, t in zip(predictions, true_labels))
        fp = sum(p == 1 and t == 0 for p, t in zip(predictions, true_labels))
        fn = sum(p == 0 and t == 1 for p, t in zip(predictions, true_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return total_loss / len(dataloader), accuracy, f1
