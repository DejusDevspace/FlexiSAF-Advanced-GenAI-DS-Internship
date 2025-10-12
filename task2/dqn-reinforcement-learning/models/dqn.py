import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network: Neural network that approximates Q-values.
    Q(s,a) = Expected future reward for taking action 'a' in state 's'
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()

        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Forward pass: state → Q-values for all actions
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
