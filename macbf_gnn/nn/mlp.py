import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

from .utils import init_param


class MLP(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_layers: tuple,
                 hidden_activation: nn.Module = nn.ReLU(), output_activation: nn.Module = None,
                 init: bool = True, gain: float = 1., limit_lip: bool = False):
        super().__init__()

        layers = []
        units = in_channels
        for next_units in hidden_layers:
            if init:
                if limit_lip:
                    layers.append(init_param(spectral_norm(nn.Linear(units, next_units)), gain=gain))
                else:
                    layers.append(init_param(nn.Linear(units, next_units), gain=gain))
            else:
                if limit_lip:
                    layers.append(spectral_norm(nn.Linear(units, next_units)))
                else:
                    layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        if init:
            if limit_lip:
                layers.append(init_param(spectral_norm(nn.Linear(units, out_channels)), gain=gain))
            else:
                layers.append(init_param(nn.Linear(units, out_channels), gain=gain))
        else:
            if limit_lip:
                layers.append(spectral_norm(nn.Linear(units, out_channels)))
            else:
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
