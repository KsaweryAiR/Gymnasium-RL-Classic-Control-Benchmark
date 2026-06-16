import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, enable_dueling: bool = True):
        super().__init__()

        self.enable_dueling = enable_dueling
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))

        if self.enable_dueling:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q
