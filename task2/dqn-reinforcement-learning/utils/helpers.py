import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from agent.dqn_agent import DQNAgent

def load_trained_agent(checkpoint_path, env_name):
    """
    Load a trained DQN agent from checkpoint.
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.load(checkpoint_path)
    agent.epsilon = 0  # No exploration for demo

    print(f"✓ Loaded trained agent from {checkpoint_path}")
    return agent, env
