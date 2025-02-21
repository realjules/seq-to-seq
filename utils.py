import torch
import Levenshtein
from ctcdecode import CTCBeamDecoder

def decode_prediction(log_probs, output_lengths, decoder, labels):
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(log_probs, output_lengths)
    
    predictions = []
    for beam_result, out_len in zip(beam_results[:, 0], out_lens[:, 0]):
        # Get the top beam result
        prediction = beam_result[:out_len].tolist()
        # Convert indices to characters
        prediction_str = ''.join([labels[idx] for idx in prediction])
        predictions.append(prediction_str)
    
    return predictions

def calculate_levenshtein_distance(predictions, targets):
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        distance = Levenshtein.distance(pred, target)
        total_distance += distance
        total_length += len(target)
    
    return (total_distance / total_length) * 100

class LRScheduler:
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            verbose=True
        )
    
    def step(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']