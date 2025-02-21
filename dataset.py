import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

class ASRDataset(Dataset):
    def __init__(self, data_path, feature_path):
        self.data = pd.read_csv(data_path)
        self.feature_path = feature_path
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load features from file
        features = np.load(f"{self.feature_path}/{row['id']}.npy")
        features = torch.FloatTensor(features)
        
        # Get sequence length
        seq_length = len(features)
        
        return features, seq_length

def collate_fn(batch):
    # Separate features and lengths
    features = [item[0] for item in batch]
    lengths = torch.LongTensor([item[1] for item in batch])
    
    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True)
    
    return features_padded, lengths

def create_dataloaders(train_path, val_path, feature_path, batch_size, num_workers=4):
    # Create datasets
    train_dataset = ASRDataset(train_path, feature_path)
    val_dataset = ASRDataset(val_path, feature_path)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader