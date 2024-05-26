import numpy as np
from typing import Any


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority: float, data: Any):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx: int, change: float):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value: float) -> tuple[int, float, Any]:
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0
        self.capacity = capacity

    def add(self, error: float, transition: tuple):
        priority = (error + 1e-5) ** self.alpha
        self.tree.add(priority, transition)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list, list[int], np.ndarray]:
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        sampling_probabilities = priorities / self.tree.total_priority
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        is_weight = np.array(is_weight, dtype=np.float32)
        
        return batch, idxs, is_weight

    def update_priorities(self, idxs: list[int], errors: list[float], eps: float = 1e-5):
        for idx, error in zip(idxs, errors):
            priority = (error + eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self) -> int:
        return len(self.tree.data)
