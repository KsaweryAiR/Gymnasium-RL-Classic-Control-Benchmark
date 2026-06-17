from collections import deque
import random
from typing import List, Tuple, Optional


class ReplayMemory:

    def __init__(self, maxlen: int, seed: Optional[int] = None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self, transition: Tuple) -> None:
        self.memory.append(transition)

    def sample(self, sample_size: int) -> List[Tuple]:
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)
