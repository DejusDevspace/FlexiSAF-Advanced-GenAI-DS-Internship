import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
from agent.dqn_agent import DQNAgent
from evaluation.visualizations import plot_training_curves
import torch
from collections import deque

# Config and hyperparameters
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 500
MAX_STEPS = 500
TARGET_UPDATE_FREQ = 10
SAVE_FREQ = 50


def train_dqn():
    """
    Main training loop for DQN agent.
    """
    # Create environment
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"Environment: {ENV_NAME}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64
    )

    # Training metrics
    episode_rewards = []
    episode_losses = []
    moving_avg_rewards = deque(maxlen=100)

    print("\nStarting training...")
    print("-" * 60)

    for episode in range(NUM_EPISODES):
        state, info = env.reset()

        episode_reward = 0
        episode_loss = []

        for step in range(MAX_STEPS):
            # Select action
            action = agent.select_action(state, training=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            # Check if episode has ended
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            # Update state and reward
            state = next_state
            episode_reward += reward

            if done:
                break

        # Decay exploration rate
        agent.decay_epsilon()

        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Record metrics
        episode_rewards.append(episode_reward)
        moving_avg_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(moving_avg_rewards)
            print(f"Episode {episode + 1}/{NUM_EPISODES} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (100 eps): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (episode + 1) % SAVE_FREQ == 0:
            agent.save(f'../artifacts/checkpoints/dqn_checkpoint_ep{episode + 1}.pth')
            print(f"✓ Checkpoint saved at episode {episode + 1}")

        # Early stopping if solved
        if np.mean(moving_avg_rewards) >= 195 and ENV_NAME == "CartPole-v1":
            print(f"\n🎉 Environment solved in {episode + 1} episodes!")
            print(f"Average reward over last 100 episodes: {np.mean(moving_avg_rewards):.2f}")
            break

    # Save final model
    agent.save('dqn_final_model.pth')
    print("\n✓ Training complete! Final model saved.")

    # Close environment
    env.close()

    # Plot training curves
    plot_training_curves(episode_rewards, episode_losses)

    return agent, episode_rewards

def run_agent(agent, env_name, num_episodes=10, render=False):
    """
    Test trained agent without exploration.
    """
    env = gym.make(env_name, render_mode='human' if render else None)
    test_rewards = []

    print(f"\nTesting agent for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset()

        episode_reward = 0
        done = False

        while not done:
            # Select best action (no exploration)
            action = agent.select_action(state, training=False)

            # Take action (Gymnasium API)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

    avg_reward = np.mean(test_rewards)
    print(f"\nAverage Test Reward: {avg_reward:.2f}")
    print(f"Std Dev: {np.std(test_rewards):.2f}")

    return test_rewards


if __name__ == "__main__":
    # Create checkpoint directory
    import os

    os.makedirs('../artifacts/checkpoints', exist_ok=True)

    # Train agent
    agent, rewards = train_dqn()

    # Test trained agent
    print("\n" + "=" * 60)
    print("TESTING TRAINED AGENT")
    print("=" * 60)
    run_agent(agent, ENV_NAME, num_episodes=10, render=False)
