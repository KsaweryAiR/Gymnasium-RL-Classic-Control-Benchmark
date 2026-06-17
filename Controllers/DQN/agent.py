import torch
from torch import nn
import torch.optim as optim
from typing import List, Tuple

from config import Hyperparameters
from .network import DQN
from .experience_replay import ReplayMemory

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:

    def __init__(self, num_states: int, num_actions: int, hyperparameters: Hyperparameters):
        self.hp = hyperparameters
        
        self.num_states = num_states
        self.num_actions = num_actions

        self.policy_dqn = DQN(
            self.num_states,
            self.num_actions,
            self.hp.hidden_layers,
            enable_dueling=self.hp.enable_dueling_dqn
        ).to(device)

        loss_class = getattr(nn, self.hp.loss_function)
        self.loss_fn = loss_class()
        
        self.optimizer = None

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

    def init_training(self) -> Tuple[DQN, ReplayMemory]:
        target_dqn = DQN(
            self.num_states,
            self.num_actions,
            self.hp.hidden_layers,
            enable_dueling=self.hp.enable_dueling_dqn
        ).to(device)
        
        target_dqn.load_state_dict(self.policy_dqn.state_dict())

        memory = ReplayMemory(self.hp.replay_memory_size)

        optimizer_class = getattr(optim, self.hp.optimizer)
        self.optimizer = optimizer_class(self.policy_dqn.parameters(), lr=self.hp.learning_rate)

        return target_dqn, memory

    def optimize(self, mini_batch: List[Tuple], target_dqn: DQN) -> None:
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.hp.enable_double_dqn:
                best_actions = self.policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.hp.discount_factor * \
                    target_dqn(new_states).gather(dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.hp.discount_factor * \
                    target_dqn(new_states).max(dim=1)[0]

        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target(self, target_dqn: DQN) -> None:
        target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.policy_dqn.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_dqn.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.policy_dqn.eval()