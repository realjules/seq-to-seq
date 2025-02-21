import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(ASRModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Process through LSTM
        packed_output, _ = self.lstm(packed)
        
        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Project to output classes
        logits = self.fc(output)
        
        # Apply log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs, None  # None is for output lengths if needed