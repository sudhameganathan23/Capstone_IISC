# OpenStack Log Anomaly Detection using LogBERT

This project implements an anomaly detection system for OpenStack logs using transformers and LogBERT. The system analyzes log sequences to identify potential anomalies in system behavior.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data_loader.py
├── model.py
└── train.py
```

## Components

- `data_loader.py`: Handles downloading and preprocessing of OpenStack log data
- `model.py`: Contains the LogBERT model implementation and training logic
- `train.py`: Main script for training the model
- `requirements.txt`: Lists all Python dependencies

## Features

- Downloads and processes OpenStack log data automatically
- Implements LogBERT architecture using PyTorch and Transformers
- Supports sequence-based log anomaly detection
- Includes training, validation, and evaluation capabilities
- Provides visualization of training metrics
- Saves the best model based on F1 score

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Requests
- tqdm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd openstack-logbert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python train.py
```

The script will:
1. Download and preprocess OpenStack log data
2. Train the LogBERT model
3. Generate training metrics visualization
4. Save the best model based on validation F1 score

## Model Architecture

The LogBERT model consists of:
- A BERT-based encoder for log sequence modeling
- Custom anomaly detection head
- Sequence-level classification for anomaly detection

## Training Process

The training process includes:
- Automatic data downloading and preprocessing
- Sequence creation from log entries
- Model training with validation
- Metrics tracking (loss, accuracy, F1 score)
- Best model checkpoint saving
- Training visualization

## Output

The training process generates:
- `best_model.pth`: Best model weights based on validation F1 score
- `training_metrics.png`: Visualization of training metrics

## Data Sources

The project uses OpenStack log data from the LogHub repository:
- Structured logs: OpenStack_2k.log_structured.csv
- Templates: OpenStack_2k.log_templates.csv
- Raw logs: OpenStack_2k.log

## Performance Metrics

The model's performance is evaluated using:
- Loss (Binary Cross Entropy)
- Accuracy
- F1 Score

## License

This project is licensed under the MIT License - see the LICENSE file for details.
