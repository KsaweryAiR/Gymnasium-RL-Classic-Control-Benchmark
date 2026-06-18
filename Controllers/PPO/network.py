import torch
from torch import nn
from typing import List

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
        super().__init__()

        #ACTOR (POLICY)
        actor_layers = []
        in_dim = state_dim

        for h_dim in hidden_layers:
            actor_layers.append(nn.Linear(in_dim, h_dim))
            actor_layers.append(nn.Tanh())
            in_dim = h_dim
        
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        #CRITIC (VALUE)
        critic_layers = []
        in_dim = state_dim

        for h_dim in hidden_layers:
            critic_layers.append(nn.Linear(in_dim, h_dim))
            critic_layers.append(nn.Tanh())
            in_dim = h_dim
            
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self):
        raise NotImplementedError