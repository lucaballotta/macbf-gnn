import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_layers: tuple,
                 hidden_activation: nn.Module = nn.ReLU(), output_activation: nn.Module = None):
        super().__init__()

        layers = []
        units = in_channels
        for next_units in hidden_layers:
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
            
        layers.append(nn.Linear(units, out_channels))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class LazyMLP(nn.Module):

    def __init__(self, out_channels: int, hidden_layers: tuple,
                 hidden_activation: nn.Module = nn.ReLU(), output_activation: nn.Module = None):
        super().__init__()

        layers = [nn.LazyLinear(hidden_layers[0])]
        units = hidden_layers[0]
        for i in range(1, len(hidden_layers)):
            next_units = hidden_layers[i]
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
            
        layers.append(nn.Linear(units, out_channels))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)