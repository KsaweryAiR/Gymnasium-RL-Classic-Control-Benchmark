import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class DQN(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], enable_dueling: bool = True):
        super().__init__()

        self.enable_dueling = enable_dueling
        
        layers = []
        in_dim = state_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim 
            
        self.feature_extractor = nn.Sequential(*layers)

        if self.enable_dueling:
            self.fc_value = nn.Linear(in_dim, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantages = nn.Linear(in_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)

        if self.enable_dueling:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q
