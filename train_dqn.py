import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
import gymnasium as gym
import ale_py

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
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

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
        resized = cv2.resize(gray, (84, 84))
        return resized.astype(np.float32) / 255.0

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
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Experience replay
        self.memory = ReplayBuffer(100000)
        self.batch_size = 32
        self.update_target_every = 1000
        self.steps = 0

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.q_network.fc[-1].out_features)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay


def train_dqn():
    # Environment setup
    env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
    preprocessor = AtariPreprocessor()

    # Agent setup
    state_shape = (4, 84, 84)  # 4 stacked frames
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions)

    episodes = 1000
    scores = deque(maxlen=100)

    for episode in range(episodes):
        state = preprocessor.reset(env)
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = preprocessor.step(env, action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        avg_score = np.mean(scores)

        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, "
                  f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

        # Save model periodically
        if episode % 100 == 0:
            torch.save(agent.q_network.state_dict(), f'dqn_pacman_{episode}.pth')


if __name__ == "__main__":
    train_dqn()