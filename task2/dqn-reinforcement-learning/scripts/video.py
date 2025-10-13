import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import torch
from agent.dqn_agent import DQNAgent
import imageio
from PIL import Image, ImageDraw, ImageFont


def record_episode(env_name, agent=None, max_steps=100):
    """
    Record one episode and return frames.

    Args:
        env_name: Name of the environment
        agent: Trained agent (None for random)
        max_steps: Maximum steps per episode

    Returns:
        frames: List of RGB frames
        episode_reward: Total reward
        steps: Number of steps taken
    """
    env = gym.make(env_name, render_mode='rgb_array')
    state, info = env.reset()

    frames = []
    episode_reward = 0

    for step in range(max_steps):
        # Render frame
        frame = env.render()
        frames.append(frame)

        # Select action
        if agent is None:
            action = env.action_space.sample()  # Random
        else:
            action = agent.select_action(state, training=False)  # Trained

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        episode_reward += reward

        if done:
            break

    env.close()
    return frames, episode_reward, len(frames)


def create_side_by_side_video(
        env_name='CartPole-v1',
        checkpoint_path='dqn_final_model.pth',
        output_file='random_vs_trained.mp4',
        num_episodes=3,
        fps=30
):
    """
    Create side-by-side comparison video.

    Args:
        env_name: Gym environment name
        checkpoint_path: Path to trained model
        output_file: Output video filename
        num_episodes: Number of episodes to show
        fps: Frames per second
    """
    print("=" * 60)
    print("Creating Comparison Video")
    print("=" * 60)

    # Load trained agent
    print("\n[1/4] Loading trained agent...")
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    agent = DQNAgent(state_size, action_size)
    agent.load(checkpoint_path)
    agent.epsilon = 0
    print("✓ Trained agent loaded")

    # Collect frames from multiple episodes
    all_frames = []

    for episode in range(num_episodes):
        print(f"\n[{episode + 2}/{num_episodes + 3}] Recording episode {episode + 1}...")

        # Record random agent
        print("  Recording random agent...")
        random_frames, random_reward, random_steps = record_episode(env_name, agent=None)

        # Record trained agent
        print("  Recording trained agent...")
        trained_frames, trained_reward, trained_steps = record_episode(env_name, agent=agent)

        print(f"  Random: {random_reward:.0f} reward in {random_steps} steps")
        print(f"  Trained: {trained_reward:.0f} reward in {trained_steps} steps")

        # Match frame counts (use shorter episode length)
        min_frames = min(len(random_frames), len(trained_frames))

        # Create side-by-side frames
        for i in range(min_frames):
            # Combine frames horizontally
            combined = combine_frames(
                random_frames[i],
                trained_frames[i],
                f"Episode {episode + 1}",
                random_reward,
                trained_reward,
                i,
                min_frames
            )
            all_frames.append(combined)

        # Add pause between episodes (30 frames = 1 second at 30fps)
        if episode < num_episodes - 1:
            pause_frame = combined.copy()
            for _ in range(30):
                all_frames.append(pause_frame)

    # Save video
    print(f"\n[{num_episodes + 3}/{num_episodes + 3}] Saving video...")
    imageio.mimsave(output_file, all_frames, fps=fps)

    print("\n" + "=" * 60)
    print(f"✅ Video saved: {output_file}")
    print(f"   Total frames: {len(all_frames)}")
    print(f"   Duration: {len(all_frames) / fps:.1f} seconds")
    print("=" * 60)

def combine_frames(left_frame, right_frame, title, left_reward, right_reward, frame_num, total_frames):
    """
    Combine two frames side-by-side with labels and stats.
    """
    # Convert to PIL Images
    left_img = Image.fromarray(left_frame)
    right_img = Image.fromarray(right_frame)

    # Get dimensions
    width, height = left_img.size

    # Create combined image with space for text
    text_height = 80
    combined_width = width * 2 + 20  # 20px gap
    combined_height = height + text_height

    combined = Image.new('RGB', (combined_width, combined_height), color='white')

    # Paste frames
    combined.paste(left_img, (0, text_height))
    combined.paste(right_img, (width + 20, text_height))

    # Add text
    draw = ImageDraw.Draw(combined)

    # Try to use a nice font, fallback to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 20)
        stats_font = ImageFont.truetype("arial.ttf", 16)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        stats_font = ImageFont.load_default()

    # Title
    title_text = f"DQN Agent Comparison - {title}"
    draw.text((combined_width // 2 - 150, 10), title_text, fill='black', font=title_font)

    # Labels
    draw.text((width // 2 - 80, 45), "Random Agent", fill='red', font=label_font)
    draw.text((width + width // 2 - 80, 45), "Trained Agent", fill='green', font=label_font)

    # Rewards
    draw.text((20, 45), f"Reward: {left_reward:.0f}", fill='red', font=stats_font)
    draw.text((width + 40, 45), f"Reward: {right_reward:.0f}", fill='green', font=stats_font)

    # Progress bar
    progress = frame_num / total_frames
    bar_width = combined_width - 40
    bar_height = 10
    bar_x = 20
    bar_y = 65

    # Background
    draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], fill='lightgray')
    # Progress
    draw.rectangle([bar_x, bar_y, bar_x + int(bar_width * progress), bar_y + bar_height], fill='blue')

    return np.array(combined)

if __name__ == "__main__":
    print("\n🎬 DQN Comparison Video Creator\n")


    ENV_NAME = "CartPole-v1"
    CHECKPOINT = "../artifacts/models/dqn_final_model.pth"

    create_side_by_side_video(
        env_name=ENV_NAME,
        checkpoint_path=CHECKPOINT,
        output_file='../img/random_vs_trained.mp4',
        num_episodes=3,
        fps=30
    )

    print("\n🎉 All done! Video saved in ../img/random_vs_trained.mp4")
