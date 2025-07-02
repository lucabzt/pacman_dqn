import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
import gymnasium as gym
import ale_py
import time
import matplotlib.pyplot as plt
import os  # Added to create a directory for models


# ... (All the classes from the previous code block: DQN, ReplayBuffer, AtariPreprocessor, DQNAgent) ...
# (The code for the classes is unchanged)
class DQN(nn.Module):
    """Improved CNN architecture compared to original paper"""

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x / 255.0
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state.astype(np.float32), action, reward, next_state.astype(np.float32), done

    def __len__(self):
        return len(self.buffer)


class AtariPreprocessor:
    """Preprocessing for Atari frames"""

    def __init__(self, frame_skip=4, frame_stack=4):
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, env):
        obs, info = env.reset()
        frame = self.preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return np.array(self.frames)

    def step(self, env, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        frame = self.preprocess_frame(obs)
        self.frames.append(frame)
        return np.array(self.frames), total_reward, terminated or truncated, info


class DQNAgent:
    def __init__(self, state_shape, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_frames=1_000_000,
                 replay_buffer_capacity=1_000_000, batch_size=32,
                 update_target_every=10000, replay_start_size=50000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.replay_start_size = replay_start_size
        self.memory = ReplayBuffer(replay_buffer_capacity)
        self.training_steps = 0

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_epsilon(self, frame_idx):
        if frame_idx > self.epsilon_decay_frames:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
                    frame_idx / self.epsilon_decay_frames)

    def replay(self):
        if len(self.memory) < self.replay_start_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1]
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss.item()


def plot_and_save_rewards(frame_indices, rewards, filename="dqn_rewards_plot.png"):
    if not rewards:
        print("No episodes completed, skipping plot generation.")
        return
    plt.figure(figsize=(12, 6))
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Total Frames")
    plt.ylabel("Episode Reward")
    moving_avg = []
    for i in range(len(rewards)):
        start_idx = max(0, i - 99)
        moving_avg.append(np.mean(rewards[start_idx:i + 1]))
    plt.plot(frame_indices, rewards, label="Episode Reward", alpha=0.7, zorder=1)
    plt.plot(frame_indices, moving_avg, label="100-Episode Moving Average", color='red', linewidth=2, zorder=2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"\nReward plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()


def train_dqn():
    # --- Hyperparameters ---
    ENV_NAME = 'ALE/Pong-v5'
    TOTAL_FRAMES = 1_000_000  # Increased for more meaningful training
    REPLAY_MEMORY_CAPACITY = 100_000  # Reduced for faster start, still effective
    EPSILON_DECAY_FRAMES = 250_000
    REPLAY_START_SIZE = 10_000
    TARGET_UPDATE_FREQUENCY = 1_000  # In training steps
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    # --- New Saving Hyperparameters ---
    MODEL_SAVE_PATH = "models"
    CHECKPOINT_SAVE_FREQUENCY = 50_000  # Save a checkpoint every N frames

    # Create directory for saving models if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    env_name_simple = ENV_NAME.split("/")[-1].lower()  # e.g., 'pong-v5'

    # Environment setup
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    preprocessor = AtariPreprocessor()

    # Agent setup
    state_shape = (4, 84, 84)
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape=state_shape, n_actions=n_actions,
                     lr=LEARNING_RATE, gamma=GAMMA,
                     epsilon_decay_frames=EPSILON_DECAY_FRAMES,
                     replay_buffer_capacity=REPLAY_MEMORY_CAPACITY,
                     batch_size=BATCH_SIZE,
                     update_target_every=TARGET_UPDATE_FREQUENCY,
                     replay_start_size=REPLAY_START_SIZE)

    # --- Training loop ---
    scores = deque(maxlen=100)
    plot_episode_rewards = []
    plot_episode_frames = []

    best_avg_score = -float('inf')  # Initialize best score tracker

    state = preprocessor.reset(env)
    episode_reward = 0
    start_time = time.time()

    try:
        for frame_idx in range(1, TOTAL_FRAMES + 1):
            agent.update_epsilon(frame_idx)
            action = agent.act(state)

            next_state, reward, done, info = preprocessor.step(env, action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()

            state = next_state
            episode_reward += reward

            # --- Periodic Checkpoint Saving ---
            if frame_idx % CHECKPOINT_SAVE_FREQUENCY == 0:
                ckpt_path = os.path.join(MODEL_SAVE_PATH, f'dqn_{env_name_simple}_ckpt_{frame_idx}.pth')
                torch.save(agent.q_network.state_dict(), ckpt_path)
                print(f"--- Saved checkpoint at frame {frame_idx} to {ckpt_path} ---")

            if done:
                scores.append(episode_reward)
                plot_episode_rewards.append(episode_reward)
                plot_episode_frames.append(frame_idx)
                avg_score = np.mean(scores)

                print(f"Frame {frame_idx}/{TOTAL_FRAMES} | "
                      f"Episode {len(plot_episode_rewards)} | "
                      f"Score: {episode_reward:.2f} | "
                      f"Avg Score (100): {avg_score:.2f} | "
                      f"Best Avg: {best_avg_score:.2f} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Loss: {loss if loss is not None else 'N/A'}")

                # --- Best Model Saving Logic ---
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_model_path = os.path.join(MODEL_SAVE_PATH, f'dqn_{env_name_simple}_best.pth')
                    torch.save(agent.q_network.state_dict(), best_model_path)
                    print(f"*** New best average score: {best_avg_score:.2f}! Model saved to {best_model_path} ***")

                state = preprocessor.reset(env)
                episode_reward = 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()
        end_time = time.time()
        print(f"Training finished or interrupted after {(end_time - start_time) / 3600:.2f} hours.")
        plot_and_save_rewards(plot_episode_frames, plot_episode_rewards, f"dqn_{env_name_simple}_rewards.png")


if __name__ == "__main__":
    train_dqn()