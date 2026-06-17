import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List

matplotlib.use('Agg')


def save_graph(graph_path: str, rewards_per_episode: List[float], epsilon_history: List[float]) -> None:
    fig = plt.figure(1)

    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

    plt.subplot(121)
    plt.ylabel('Mean Rewards')
    plt.plot(mean_rewards)

    plt.subplot(122)
    plt.ylabel('Epsilon Decay')
    plt.plot(epsilon_history)

    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    fig.savefig(graph_path)
    plt.close(fig)
