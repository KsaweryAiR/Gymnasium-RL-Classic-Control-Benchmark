import torch
from torch import nn
from typing import List, Tuple

from config import Hyperparameters
from dqn_controller import DQN
from experience_replay import ReplayMemory

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:

    def __init__(self, num_states: int, num_actions: int, hyperparameters: Hyperparameters):
        self.hp = hyperparameters

        self.policy_dqn = DQN(
            num_states,
            num_actions,
            self.hp.fc1_nodes,
            enable_dueling=self.hp.enable_dueling_dqn
        ).to(device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

    def init_training(self) -> Tuple[DQN, ReplayMemory]:
        target_dqn = DQN(
            self.policy_dqn.fc1.in_features,
            self.policy_dqn.advantages.out_features if self.hp.enable_dueling_dqn else self.policy_dqn.output.out_features,
            self.hp.fc1_nodes,
            enable_dueling=self.hp.enable_dueling_dqn
        ).to(device)
        target_dqn.load_state_dict(self.policy_dqn.state_dict())

        memory = ReplayMemory(self.hp.replay_memory_size)

        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.hp.learning_rate)

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
                # Standardowy DQN: target_dqn wybiera i ocenia akcję – może prowadzić do przeszacowania Q
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
