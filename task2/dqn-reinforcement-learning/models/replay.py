import random
import torch
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer: Stores past experiences for training.
    Breaks correlation between consecutive experiences.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample random batch of experiences.
        Random sampling breaks temporal correlations.
        """
        experiences = random.sample(self.buffer, batch_size)

        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
