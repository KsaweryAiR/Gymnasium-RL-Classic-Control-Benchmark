import torch
import gymnasium as gym
import itertools
import os
import flappy_bird_gymnasium

from datetime import datetime, timedelta
from typing import Optional
from Controllers.PPO.agent import Agent
from Controllers.PPO.rollout_buffer import RolloutBuffer
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

        self.log_file   = os.path.join(RUNS_DIR, f'{self.set_name}_PPO.log')
        self.model_file = os.path.join(RUNS_DIR, f'{self.set_name}_PPO.pt')
        self.graph_file = os.path.join(RUNS_DIR, f'{self.set_name}_PPO.png')

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
        self._log(f"{start_time.strftime(DATE_FORMAT)}: Rozpoczynamy trening PPO...", mode='w')

        env = self._make_env(render=False)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        agent = Agent(num_states, num_actions, self.hp)
        buffer = RolloutBuffer()

        rewards_per_episode = []
        best_avg = float('-inf')  
        last_graph_update = start_time

        max_timesteps = getattr(self.hp, 'total_timesteps', None)
        stop_reward = getattr(self.hp, 'stop_on_reward', None)

        for global_step in itertools.count(1):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                
                action, value, log_prob = agent.select_action(state)

                new_state_np, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state_np, dtype=torch.float, device=device)
                
                buffer.store(state, action, reward, value, log_prob, terminated or truncated)
                
                state = new_state
                global_step += 1

                if len(buffer.states) >= self.hp.n_steps:
                    next_value = agent.get_value(state)
                    agent.optimize(buffer, next_value)
                    buffer.clear()

                if max_timesteps is not None and global_step >= max_timesteps:
                    self._log(f"Trening przerwany: Osiągnięto limit {max_timesteps} kroków.")
                    self._end_training(env, rewards_per_episode)
                    return

            rewards_per_episode.append(episode_reward)

            if len(rewards_per_episode) >= 100:
                avg_100 = sum(rewards_per_episode[-100:]) / 100
                if avg_100 > best_avg:
                    agent.save(self.model_file)
                    best_avg = avg_100
                    self._log(f"{datetime.now().strftime(DATE_FORMAT)}: Nowy rekord avg100: {avg_100:.1f} w epizodzie {len(rewards_per_episode)}")
                
                if stop_reward is not None and avg_100 >= stop_reward:
                    self._log(f"Środowisko rozwiązane! Osiągnięto średnią {avg_100:.1f} (Cel: {stop_reward})")
                    self._end_training(env, rewards_per_episode)
                    return

            if len(rewards_per_episode) % 10 == 0:
                avg_10 = sum(rewards_per_episode[-10:]) / 10
                # self._log(f"Krok: {global_step} | Epizod: {len(rewards_per_episode)} | Średnia (10): {avg_10:.2f}")

            if datetime.now() - last_graph_update > timedelta(seconds=10):
                dummy_epsilon = [0.0] * len(rewards_per_episode)
                save_graph(self.graph_file, rewards_per_episode, dummy_epsilon)
                last_graph_update = datetime.now()

    def _end_training(self, env, rewards_per_episode):
        self._log("Trening zakończony pomyślnie!")
        dummy_epsilon = [0.0] * len(rewards_per_episode)
        save_graph(self.graph_file, rewards_per_episode, dummy_epsilon)
        env.close()

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
                action, _, _ = agent.select_action(state)
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                state = torch.tensor(new_state, dtype=torch.float, device=device)

            print(f"Episode {episode}: reward = {episode_reward:.1f}")