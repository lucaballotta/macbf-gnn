import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, PackedSequence
from numpy import arange

class Predictor(nn.Module):
    def __init__(self, edge_dim: int, num_layers: int=1):
        super().__init__()
        self.predictor = nn.LSTM(
            input_size=edge_dim + 1, hidden_size=edge_dim, num_layers=num_layers, batch_first=True
        )
        
    def forward(self, edge_attr):
        if isinstance(edge_attr, PackedSequence):
            input = edge_attr
        else:
            input = pack_sequence(edge_attr, enforce_sorted=False)
            
        output, _ = self.predictor(input)
        padded_output = pad_packed_sequence(output, batch_first=True)
        pred = padded_output[0]
        lengths = padded_output[1]
        pred_last = pred[arange(len(pred)), lengths - 1]
        
        return pred_last