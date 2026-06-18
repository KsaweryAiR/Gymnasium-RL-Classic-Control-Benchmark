import random
import torch
import gymnasium as gym
import itertools
import os

from datetime import datetime, timedelta
from typing import Optional

from Controllers.DQN.agent import Agent
from config import Hyperparameters
from visualizer import save_graph

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:

    def __init__(self, hyperparameter_set: str, hyperparameters: Hyperparameters):
        self.hp = hyperparameters
        self.set_name = hyperparameter_set

        self.log_file   = os.path.join(RUNS_DIR, f'{self.set_name}.log')
        self.model_file = os.path.join(RUNS_DIR, f'{self.set_name}.pt')
        self.graph_file = os.path.join(RUNS_DIR, f'{self.set_name}.png')

    def _log(self, message: str, mode: str = 'a') -> None:
        print(message)
        with open(self.log_file, mode) as f:
            f.write(message + '\n')

    def _make_env(self, render: bool) -> gym.Env:
        return gym.make(
            self.hp.env_id,
            render_mode='human' if render else None,
            **self.hp.env_make_params
        )

    def train(self) -> None:
        start_time = datetime.now()
        self._log(f"{start_time.strftime(DATE_FORMAT)}: Training starting...", mode='w')

        env = self._make_env(render=False)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        agent = Agent(num_states, num_actions, self.hp)
        target_dqn, memory = agent.init_training()

        epsilon = self.hp.epsilon_init
        epsilon_history = []
        rewards_per_episode = []
        step_count = 0
        best_reward = float('-inf')
        last_graph_update = start_time

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated) and episode_reward < self.hp.stop_on_reward:
                if random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
                else:
                    action = agent.select_action(state)

                new_state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                memory.append((state, action, new_state, reward, terminated))
                step_count += 1
                state = new_state

                if len(memory) > self.hp.mini_batch_size:
                    mini_batch = memory.sample(self.hp.mini_batch_size)
                    agent.optimize(mini_batch, target_dqn)

                    if step_count > self.hp.network_sync_rate:
                        agent.sync_target(target_dqn)
                        step_count = 0

            rewards_per_episode.append(episode_reward)

            if len(memory) > self.hp.mini_batch_size:
                epsilon = max(epsilon * self.hp.epsilon_decay, self.hp.epsilon_min)
            epsilon_history.append(epsilon)

            if episode_reward > best_reward:
                msg = (
                    f"{datetime.now().strftime(DATE_FORMAT)}: "
                    f"New best reward {episode_reward:.1f} "
                    f"({(episode_reward - best_reward) / abs(best_reward) * 100:+.1f}%) "
                    f"at episode {episode}, saving model..."
                )
                self._log(msg)
                agent.save(self.model_file)
                best_reward = episode_reward

            if datetime.now() - last_graph_update > timedelta(seconds=10):
                save_graph(self.graph_file, rewards_per_episode, epsilon_history)
                last_graph_update = datetime.now()

    def test(self, render: bool = True) -> None:
        env = self._make_env(render=render)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        agent = Agent(num_states, num_actions, self.hp)
        agent.load(self.model_file)

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                action = agent.select_action(state)
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                state = torch.tensor(new_state, dtype=torch.float, device=device)

            print(f"Episode {episode}: reward = {episode_reward:.1f}")