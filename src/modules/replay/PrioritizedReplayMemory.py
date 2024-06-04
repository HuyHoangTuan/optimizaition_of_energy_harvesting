from src.modules.replay.ReplayMemory import Transition
from src.utils import RandomUtils
import numpy as np
import torch



class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.data_pointer = 0
        self.real_size = 0

    def add(self, priority, data):
        # tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(self.data_pointer, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.real_size = min(self.capacity, self.real_size + 1)

    def update(self, idx, priority):
        tree_index = idx + self.capacity - 1
        change = priority - self.tree[tree_index]
        # print(f'idx: {idx}, priority: {priority}, tree_index: {tree_index}, change: {change}')
        # print(f'{priority} {change} {self.tree[tree_index]}')
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)
        # print(f'{self.tree[0]}, {change}')

    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def get_leaf(self, v, is_trace = False):
        parent = 0
        while 2 * parent + 1 < len(self.tree):

            left_child, right_child = 2 * parent + 1, 2 * parent + 2

            # if is_trace:
            #     print(f'v: {v}, parent: {parent} - {self.tree[parent]}, left_child: {left_child} - {self.tree[left_child]} , right_child: {right_child} - {self.tree[right_child]}')

            if self.tree[left_child] == 0 and self.tree[right_child] == 0:
                break

            if v <= self.tree[left_child] or self.tree[right_child] == 0:
                v = self.tree[left_child] if v > self.tree[left_child] and self.tree[right_child] == 0 else v
                parent = left_child
            else:
                v -= self.tree[left_child]
                if v > self.tree[right_child]:
                    v = self.tree[right_child]

                parent = right_child


        data_index = parent - self.capacity + 1
        # 14687 + 100000 - 1 = 114686

        priority = self.tree[parent]

        return data_index, priority, self.data[data_index]

    def total_priority(self):
        return self.tree[0]

    def get_zero_tree(self):
        for value in self.tree:
            if value == 0:
                return self.real_size
        return None

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        # print(capacity)
        self.tree = SumTree(capacity)

        self.alpha = alpha
        self.epsilon = np.array([1e-2])  # small amount to avoid zero priority
        self.size = capacity
        self.real_size = 0
        self.max_priority = np.array([1e-2])

    def push(self, *args):
        """Save a transition with the maximum priority initially."""
        self.tree.add(self.max_priority, Transition(*args))
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions with priority-based probabilities."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            v = RandomUtils.uniform(i * segment, (i + 1) * segment)

            # if i == batch_size - 1:
            #     v = self.tree.total_priority() - self.epsilon
            _v = np.copy(v)
            leaf, priority, data = self.tree.get_leaf(v)
            batch.append(data)
            indices.append(leaf)
            priorities.append(priority)

        # print(self.tree.get_zero_tree())
        # print(_v)
        # print(self.tree.total_priority())
        # print(segment)
        # print(indices)
        # _ = self.tree.get_leaf(_v, is_trace=True)
        probabilities = np.array(priorities) / self.tree.total_priority()
        weights = (self.real_size * probabilities) ** (-beta)
        # print(probabilities)
        # print(self.tree.total_priority())
        # print(self.max_priority)
        weights /= weights.max()
        return batch, indices, weights

    def update_priorities(self, indices, priorities):

        """Update priorities for the sampled transitions."""
        for idx, priority in zip(indices, priorities):

            _priority = (priority + self.epsilon) ** self.alpha
            # print(_priority)
            # print(f'idx: {idx}, priority: {priority}')
            # self.tree.update(idx, _priority)

            self.max_priority = max(self.max_priority, _priority)
        # print(f'priority: {self.max_priority}')

    def __len__(self):
        return self.real_size