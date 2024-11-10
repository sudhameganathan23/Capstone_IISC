import pandas as pd
import requests
from typing import Tuple

def download_log_data() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Download OpenStack log data from GitHub
    Returns:
        Tuple containing structured logs DataFrame, templates DataFrame, and raw logs
    """
    # URLs for the log data
    structured_url = "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/OpenStack/OpenStack_2k.log_structured.csv"
    templates_url = "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/OpenStack/OpenStack_2k.log_templates.csv"
    raw_logs_url = "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/OpenStack/OpenStack_2k.log"
    
    try:
        # Download the data
        structured_logs = pd.read_csv(structured_url)
        templates = pd.read_csv(templates_url)
        raw_logs = requests.get(raw_logs_url).text
        
        print("Structured logs columns:", structured_logs.columns.tolist())
        print("Templates columns:", templates.columns.tolist())
        
        # Print first few templates for debugging
        print("\nFirst few log templates:")
        for i, template in enumerate(templates['EventTemplate'].head(3)):
            print(f"Template {i}: {template}")
        
        return structured_logs, templates, raw_logs
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        raise

def preprocess_logs(structured_logs: pd.DataFrame, templates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the log data for model training
    Args:
        structured_logs: DataFrame containing structured logs
        templates: DataFrame containing log templates
    Returns:
        Processed DataFrames ready for model training
    """
    # Combine Date and Time columns to create Timestamp
    structured_logs['Timestamp'] = pd.to_datetime(structured_logs['Date'] + ' ' + structured_logs['Time'])
    
    # Sort by timestamp
    structured_logs = structured_logs.sort_values('Timestamp')
    
    # Merge templates with structured logs
    merged_logs = structured_logs.merge(templates, on='EventTemplate', how='left')
    
    # Create sequential IDs for each unique template
    template_mapping = {template: idx for idx, template in enumerate(templates['EventTemplate'].unique())}
    merged_logs['TemplateId'] = merged_logs['EventTemplate'].map(template_mapping)
    
    # Print first few merged logs for debugging
    print("\nFirst few merged logs:")
    print(merged_logs[['Timestamp', 'EventTemplate', 'TemplateId']].head(3))
    
    return merged_logs, templates

def create_sequences(df: pd.DataFrame, sequence_length: int = 10) -> Tuple[list, list]:
    """
    Create sequences of log templates for training
    Args:
        df: Processed DataFrame containing log data
        sequence_length: Length of sequences to create
    Returns:
        Lists of sequences and their corresponding labels
    """
    sequences = []
    labels = []
    
    for i in range(len(df) - sequence_length + 1):
        sequence = df['TemplateId'].iloc[i:i + sequence_length].tolist()
        # Use 'Label' column if it exists, otherwise assume normal (0)
        label = 1 if 'Label' in df.columns and df['Label'].iloc[i:i + sequence_length].any() else 0
        sequences.append(sequence)
        labels.append(label)
    
    # Print first sequence for debugging
    if sequences:
        print("\nFirst sequence:")
        print(f"Template IDs: {sequences[0]}")
        print(f"Label: {labels[0]}")
    
    return sequences, labels
