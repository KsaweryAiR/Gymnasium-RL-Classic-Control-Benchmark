import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple

from config import Hyperparameters
from .network import ActorCritic
from .rollout_buffer import RolloutBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:

    def __init__(self, num_states: int, num_actions: int, hyperparameters: Hyperparameters):
        self.hp = hyperparameters
        self.num_states = num_states
        self.num_actions = num_actions

        self.policy = ActorCritic(self.num_states, self.num_actions, self.hp.hidden_layers).to(device)
        
        self.optimizer_actor = optim.Adam(self.policy.actor.parameters(), lr=self.hp.learning_rate_actor)
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=self.hp.learning_rate_critic)
        
        self.mse_loss = nn.MSELoss()


        self.gae_lambda = getattr(self.hp, 'gae_lambda', 0.95)
        self.entropy_coef = getattr(self.hp, 'entropy_coef', 0.01)
        self.critic_coef = getattr(self.hp, 'critic_loss_coef', 0.5)
        self.max_grad_norm = getattr(self.hp, 'max_grad_norm', 0.5)

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.policy.actor(state.unsqueeze(dim=0))
            value = self.policy.critic(state.unsqueeze(dim=0))
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.squeeze(), value.squeeze(), log_prob.squeeze()

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.critic(state.unsqueeze(dim=0)).squeeze()

    def optimize(self, buffer: RolloutBuffer, next_value: torch.Tensor) -> None:
        states, actions, rewards, values, log_probs, dones = buffer.get_data()
        
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        values = values.to(device)
        log_probs = log_probs.to(device)
        dones = dones.to(device)
        next_value = next_value.to(device)

        advantages = torch.zeros_like(rewards).to(device)
        last_gae = 0
        
        all_values = torch.cat([values, next_value.view(-1)])
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.hp.discount_factor * all_values[t + 1] * mask - all_values[t]
            gae = delta + self.hp.discount_factor * self.gae_lambda * mask * last_gae
            advantages[t] = gae
            last_gae = gae
            
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        
        for _ in range(self.hp.n_epochs):
            permutation = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.hp.batch_size):
                batch_indices = permutation[start_idx:start_idx + self.hp.batch_size]
                
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_log_probs = log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]
                
                logits = self.policy.actor(b_states)
                new_values = self.policy.critic(b_states).squeeze()
                
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratios = torch.exp(new_log_probs - b_log_probs)
                
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.hp.clip_range, 1.0 + self.hp.clip_range) * b_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(new_values, b_returns)
                
                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy
                
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                
                self.optimizer_actor.step()
                self.optimizer_critic.step()

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.policy.eval()
