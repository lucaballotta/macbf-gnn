import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, PackedSequence

from macbf_gnn.nn import MLP

    
class Predictor(nn.Module):
    '''def __init__(self, input_dim: int, output_dim: int, num_layers: int=4, hidden_size: int=64):
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_dim)'''
        
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, hidden_size: int = 256, device: torch.device = torch.device('cuda')):
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers, batch_first=True
        )
        self.mlp = MLP(
            in_channels=hidden_size, out_channels=output_dim, hidden_layers=(32, 32), limit_lip=True
        )
        self.device = device

    def forward(self, edge_attr):
        if isinstance(edge_attr, PackedSequence):
            input_data = edge_attr
        else:
            input_data = pack_sequence(edge_attr, enforce_sorted=False)
            
        h0 = torch.zeros(self.num_layers, input_data.batch_sizes[0], self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, input_data.batch_sizes[0], self.hidden_size, device=self.device)
        out_packed, _ = self.lstm(input_data, (h0, c0))
        out_padded = pad_packed_sequence(out_packed, batch_first=True)
        out = out_padded[0]
        lens = out_padded[1]
        out_nnz = torch.stack([sample[lens[idx] - 1, :] for idx, sample in enumerate(out)])
        pred = self.mlp(out_nnz)
        
        return pred