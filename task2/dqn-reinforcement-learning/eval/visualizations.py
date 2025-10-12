import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import imageio
from utils.helpers import load_trained_agent

def plot_training_curves(rewards, losses):
    """
    Plot training metrics: rewards and losses over episodes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot rewards
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')

    # Calculate and plot moving average
    moving_avg = []
    window = 100
    for i in range(len(rewards)):
        if i < window:
            moving_avg.append(np.mean(rewards[:i + 1]))
        else:
            moving_avg.append(np.mean(rewards[i - window:i]))

    ax1.plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot losses
    ax2.plot(losses, alpha=0.6, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Loss')
    ax2.set_title('Training Loss Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("✓ Training curves saved as '../img/training_curves.png'")
    plt.show()

def visualize_q_values(agent, env_name):
    """
    Visualize Q-values for different states (works best for CartPole).
    Shows what actions agent prefers in different situations.
    """
    env = gym.make(env_name)
    state, info = env.reset()

    # Collect Q-values for multiple states
    states_list = []
    q_values_list = []

    for _ in range(50):
        action = agent.select_action(state, training=False)

        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor).cpu().numpy()[0]

        states_list.append(state.copy())
        q_values_list.append(q_values)

        # Take step (Gymnasium API)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            state, info = env.reset()
        else:
            state = next_state

    env.close()

    # Plot Q-values
    q_values_array = np.array(q_values_list)

    plt.figure(figsize=(12, 6))
    for action in range(q_values_array.shape[1]):
        plt.plot(q_values_array[:, action], label=f'Action {action}', alpha=0.7)

    plt.xlabel('State Sample')
    plt.ylabel('Q-Value')
    plt.title('Q-Values Across Different States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../img/q_values_visualization.png', dpi=150)
    print("✓ Q-values visualization saved as 'q_values_visualization.png'")
    plt.show()

def compare_random_vs_trained(env_name, checkpoint_path, num_episodes=10):
    """
    Compare random agent vs trained agent.
    Great for showing learning effectiveness!
    """
    env = gym.make(env_name)

    # Load trained agent
    agent, _ = load_trained_agent(checkpoint_path, env_name)

    # Test random agent
    print("\n--- Testing Random Agent ---")
    random_rewards = []
    for ep in range(num_episodes):
        state, info = env.reset()

        episode_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

        random_rewards.append(episode_reward)

    # Test trained agent
    print("\n--- Testing Trained Agent ---")
    trained_rewards = []
    for ep in range(num_episodes):
        state, info = env.reset()

        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

        trained_rewards.append(episode_reward)

    env.close()

    # Plot comparison
    plot_comparison(random_rewards, trained_rewards)

    # Print statistics
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"Random Agent:  Mean = {np.mean(random_rewards):.2f}, Std = {np.std(random_rewards):.2f}")
    print(f"Trained Agent: Mean = {np.mean(trained_rewards):.2f}, Std = {np.std(trained_rewards):.2f}")
    print(
        f"Improvement: {np.mean(trained_rewards) - np.mean(random_rewards):.2f} (+{((np.mean(trained_rewards) / np.mean(random_rewards) - 1) * 100):.1f}%)")


def plot_comparison(random_rewards, trained_rewards):
    """
    Create bar chart comparing random vs trained agent.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Episode-by-episode comparison
    x = range(1, len(random_rewards) + 1)
    ax1.plot(x, random_rewards, 'o-', label='Random Agent', alpha=0.7, linewidth=2)
    ax1.plot(x, trained_rewards, 's-', label='Trained Agent', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards: Random vs Trained')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot comparison
    ax2.boxplot([random_rewards, trained_rewards], labels=['Random', 'Trained'])
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Distribution Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../img/random_vs_trained_comparison.png', dpi=150)
    print("✓ Comparison plot saved as 'img/random_vs_trained_comparison.png'")
    plt.show()

def record_episode_gif(agent, env, filename='../img/agent_demo.gif', max_steps=500):
    """
    Record agent playing and save as GIF.
    Perfect for presentations!
    """
    frames = []
    state, info = env.reset()

    episode_reward = 0

    for step in range(max_steps):
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # Agent selects action
        action = agent.select_action(state, training=False)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        episode_reward += reward

        if done:
            break

    # Save as GIF
    if frames:
        imageio.mimsave(filename, frames, fps=30)
        print(f"✓ Saved demo as '{filename}'")
        print(f"Episode Reward: {episode_reward:.2f}")

    return episode_reward
