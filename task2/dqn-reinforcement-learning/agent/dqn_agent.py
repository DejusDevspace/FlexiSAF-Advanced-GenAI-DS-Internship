import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import torch
import torch.nn as nn
import torch.optim as optim
from models.dqn import DQN
from models.replay import ReplayBuffer


class DQNAgent:
    """
    DQN Agent: Learns to play CartPole using Deep Q-Learning.
    """

    def __init__(
            self,
            state_size,
            action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon_start  # Exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Q-Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)  # Main network
        self.target_net = DQN(state_size, action_size).to(self.device)  # Target network
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights
        self.target_net.eval()  # Target network not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        - With probability epsilon: random action (exploration)
        - Otherwise: best action from Q-network (exploitation)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)
        else:
            # Exploit: best action according to policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step using experience replay.
        """
        # Need enough experiences to sample a batch
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values: Q(s,a) from policy network
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values: r + γ max Q(s',a') from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss: Mean Squared Error between current and target Q-values
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (to prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Copy weights from policy network to target network.
        Target network provides stable Q-value targets.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """
        Decay exploration rate over time.
        Agent explores more at start, exploits learned policy later.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
