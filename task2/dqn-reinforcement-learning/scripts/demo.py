import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from utils.helpers import load_trained_agent

def run_demo(env_name: str, checkpoint: str):
    """
    Run a demo of the trained DQN agent in the environment.
    """
    print("=" * 60)
    print("RUNNING DEMO SCRIPT")
    print("=" * 60)
    agent, _ = load_trained_agent(CHECKPOINT, ENV_NAME)
    env = gym.make(ENV_NAME, render_mode='human')

    print("\nWatching trained agent play...")
    print("Close the window to stop.\n")

    for episode in range(5):
        state, info = env.reset()

        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    # Configuration
    ENV_NAME = "CartPole-v1"
    CHECKPOINT = "../artifacts/models/dqn_final_model.pth"

    run_demo(ENV_NAME, CHECKPOINT)
