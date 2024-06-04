
from collections import namedtuple, deque

from src.utils import RandomUtils
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample_sequence(self, batch_size, sequence_length):
        batch = []

        for _ in range(batch_size):
            if len(self.memory) < sequence_length:
                return []
            index = RandomUtils.sample(range(len(self.memory) - sequence_length + 1), 1)[0]
            batch.append(list(self.memory)[index:index + sequence_length])

        return batch

    def sample(self, batch_size):
        return RandomUtils.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)