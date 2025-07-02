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

# Register Atari environments
gym.register_envs(ale_py)


class DQN(nn.Module):
    """Improved CNN architecture compared to original paper"""

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # More modern CNN architecture
        self.conv = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Third conv block
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            # Additional conv layer for better feature extraction
            # This layer was removed to match the classic DQN conv output size for a (84, 84) input
            # nn.Conv2d(64, 128, kernel_size=3, stride=1),
            # nn.ReLU(),
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            # Dropout is not typically used in classic DQN FC layers
            # nn.Dropout(0.1),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Normalize pixel values
        x = x / 255.0
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store states as uint8 to save memory
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        # Convert states back to float for tensor conversion
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
        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized  # Return as uint8

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

        # Networks
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Hyperparameters
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.replay_start_size = replay_start_size

        # Experience replay
        self.memory = ReplayBuffer(replay_buffer_capacity)
        self.training_steps = 0

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            # Add batch dimension and convert to tensor
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_epsilon(self, frame_idx):
        """Linearly anneal epsilon from start to end over decay_frames."""
        if frame_idx > self.epsilon_decay_frames:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
                        frame_idx / self.epsilon_decay_frames)

    def replay(self):
        # Start training only when the buffer is large enough
        if len(self.memory) < self.replay_start_size:
            return None  # Return None to indicate no loss was computed

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN logic for selecting next action
        next_actions = self.q_network(next_states).max(1)[1].detach()
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Clip grads for stability
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


def train_dqn():
    # --- Hyperparameters ---
    ENV_NAME = 'ALE/Pong-v5'
    TOTAL_FRAMES = 100_000
    REPLAY_MEMORY_CAPACITY = 1_000_000
    EPSILON_DECAY_FRAMES = 50_000
    REPLAY_START_SIZE = 10_000
    TARGET_UPDATE_FREQUENCY = 10_000  # In training steps, not env frames
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    GAMMA = 0.99

    # Environment setup
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    preprocessor = AtariPreprocessor()

    # Agent setup
    state_shape = (4, 84, 84)  # 4 stacked frames
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape=state_shape,
                     n_actions=n_actions,
                     lr=LEARNING_RATE,
                     gamma=GAMMA,
                     epsilon_decay_frames=EPSILON_DECAY_FRAMES,
                     replay_buffer_capacity=REPLAY_MEMORY_CAPACITY,
                     batch_size=BATCH_SIZE,
                     update_target_every=TARGET_UPDATE_FREQUENCY,
                     replay_start_size=REPLAY_START_SIZE)

    # Training loop
    episode_rewards = []
    scores = deque(maxlen=100)

    state = preprocessor.reset(env)
    episode_reward = 0
    start_time = time.time()

    for frame_idx in range(1, TOTAL_FRAMES + 1):
        agent.update_epsilon(frame_idx)
        action = agent.act(state)

        next_state, reward, done, info = preprocessor.step(env, action)
        agent.remember(state, action, reward, next_state, done)

        loss = agent.replay()

        state = next_state
        episode_reward += reward

        if done:
            scores.append(episode_reward)
            episode_rewards.append(episode_reward)
            avg_score = np.mean(scores)

            print(f"Frame {frame_idx}/{TOTAL_FRAMES} | "
                  f"Episode {len(episode_rewards)} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Avg Score (last 100): {avg_score:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {loss if loss is not None else 'N/A'}")

            state = preprocessor.reset(env)
            episode_reward = 0

        # Save model periodically
        if frame_idx % 250_000 == 0:
            print(f"--- Saving model at frame {frame_idx} ---")
            torch.save(agent.q_network.state_dict(), f'dqn_pacman_{frame_idx}.pth')

    env.close()
    end_time = time.time()
    print(f"Training finished in {(end_time - start_time) / 3600:.2f} hours.")


if __name__ == "__main__":
    train_dqn()
