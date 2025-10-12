import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from agent.dqn_agent import DQNAgent
from utils.helpers import load_trained_agent
from eval.visualizations import compare_random_vs_trained, visualize_q_values
import imageio

def perform_eval(env_name='CartPole-v1', checkpoint_path='dqn_final_model.pth'):
    """
    Perform and save evaluation of for a trained DQN agent.
    Compares trained agent vs random, visualizes Q-values, and records a demo GIF.
    """
    print("=" * 60)
    print("RUNNING DEMO PRESENTATION SCRIPT")
    print("=" * 60)

    # Load agent
    agent, env = load_trained_agent(checkpoint_path, env_name)

    # Compare with random agent
    print("\n[1/3] Comparing with random agent...")
    compare_random_vs_trained(env_name, checkpoint_path, num_episodes=10)

    # Visualize Q-values
    print("\n[2/3] Visualizing Q-values...")
    visualize_q_values(agent, env_name)

    # Record demo GIF
    print("\n[3/3] Recording demo GIF...")
    try:
        env_render = gym.make(env_name, render_mode='rgb_array')
        state, info = env_render.reset()

        frames = []
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 500:
            frame = env_render.render()
            frames.append(frame)

            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env_render.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            step_count += 1

        env_render.close()

        # Save GIF
        imageio.mimsave('../img/trained_agent_demo.gif', frames, fps=30)
        print(f"✓ Demo GIF saved! Episode reward: {episode_reward:.2f}")

    except Exception as e:
        print(f"⚠ Could not create GIF: {e}")

    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • img/random_vs_trained_comparison.png")
    print("  • img/q_values_visualization.png")
    print("  • img/trained_agent_demo.gif")


if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    CHECKPOINT = "../artifacts/models/dqn_final_model.pth"

    print("DQN Agent Demo")
    perform_eval(ENV_NAME, CHECKPOINT)
