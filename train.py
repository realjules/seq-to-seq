import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
import wandb

from model import ASRModel
from dataset import create_dataloaders
from utils import decode_prediction, calculate_levenshtein_distance, LRScheduler

def train(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ASRModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path=config['train_path'],
        val_path=config['val_path'],
        feature_path=config['feature_path'],
        batch_size=config['batch_size']
    )
    
    # Setup training
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = LRScheduler(optimizer, patience=config['patience'])
    
    # Create decoder for validation
    decoder = CTCBeamDecoder(
        labels=config['labels'],
        beam_width=config['beam_width'],
        log_probs_input=True
    )
    
    # Training loop
    best_val_dist = float('inf')
    for epoch in range(config['epochs']):
        print(f"\nEpoch: {epoch+1}/{config['epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0
        for x, lx in tqdm(train_loader):
            x = x.to(device)
            
            # Forward pass
            log_probs, _ = model(x, lx)
            
            # Calculate loss
            loss = criterion(
                log_probs.transpose(0, 1),
                targets,
                input_lengths=lx,
                target_lengths=ly
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, lx in tqdm(val_loader):
                x = x.to(device)
                
                # Forward pass
                log_probs, _ = model(x, lx)
                
                # Decode predictions
                predictions = decode_prediction(log_probs, lx, decoder, config['labels'])
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # Calculate loss
                loss = criterion(
                    log_probs.transpose(0, 1),
                    targets,
                    input_lengths=lx,
                    target_lengths=ly
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_dist = calculate_levenshtein_distance(all_predictions, all_targets)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        print(f"\tTrain Loss {train_loss:.4f}\t Learning Rate {scheduler.get_last_lr():.7f}")
        print(f"\tVal Dist {val_dist:.4f}%\t Val Loss {val_loss:.4f}")
        
        # Save checkpoints
        torch.save(model.state_dict(), f"{config['save_dir']}/epoch_{epoch+1}.pt")
        print("Saved epoch model")
        
        if val_dist < best_val_dist:
            best_val_dist = val_dist
            torch.save(model.state_dict(), f"{config['save_dir']}/best_model.pt")
            print("Saved best model")

if __name__ == "__main__":
    config = {
        'input_dim': 40,  # Feature dimension
        'hidden_dim': 512,  # LSTM hidden dimension
        'num_layers': 4,  # Number of LSTM layers
        'num_classes': 29,  # Number of output classes
        'dropout': 0.1,  # Dropout rate
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 100,
        'patience': 10,
        'beam_width': 10,
        'labels': ['-'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ\'_ '),  # CTC labels
        'train_path': 'data/train.csv',
        'val_path': 'data/val.csv',
        'feature_path': 'data/features',
        'save_dir': 'checkpoints'
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Start training
    train(config)