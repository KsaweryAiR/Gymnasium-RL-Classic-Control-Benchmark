import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Hyperparameters:
    env_id: str
    learning_rate: float
    discount_factor: float
    network_sync_rate: int
    replay_memory_size: int
    mini_batch_size: int
    epsilon_init: float
    epsilon_decay: float
    epsilon_min: float
    stop_on_reward: float
    fc1_nodes: int
    enable_double_dqn: bool
    enable_dueling_dqn: bool
    env_make_params: dict = field(default_factory=dict)


def load_hyperparameters(path: str, hyperparameter_set: str) -> Hyperparameters:
    with open(path, 'r') as f:
        all_sets = yaml.safe_load(f)

    if hyperparameter_set not in all_sets:
        raise KeyError(f"Hyperparameter set '{hyperparameter_set}' not found in {path}")

    params = all_sets[hyperparameter_set]

    return Hyperparameters(
        env_id=params['env_id'],
        learning_rate=params['learning_rate_a'],
        discount_factor=params['discount_factor_g'],
        network_sync_rate=params['network_sync_rate'],
        replay_memory_size=params['replay_memory_size'],
        mini_batch_size=params['mini_batch_size'],
        epsilon_init=params['epsilon_init'],
        epsilon_decay=params['epsilon_decay'],
        epsilon_min=params['epsilon_min'],
        stop_on_reward=params['stop_on_reward'],
        fc1_nodes=params['fc1_nodes'],
        enable_double_dqn=params.get('enable_double_dqn', False),
        enable_dueling_dqn=params.get('enable_dueling_dqn', False),
        env_make_params=params.get('env_make_params', {}),
    )
