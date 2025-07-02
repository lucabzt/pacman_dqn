import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
import time
import argparse
from collections import deque
import imageio
import ale_py

gym.register_envs(ale_py)



# Import the DQN class from the training script
class DQN(nn.Module):
    """Same architecture as training script"""

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

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


class AtariPreprocessor:
    """Same preprocessing as training"""

    def __init__(self, frame_skip=4, frame_stack=4):
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
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


class GameplayTester:
    def __init__(self, model_path, device='auto'):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'auto' else "cpu")
        print(f"Using device: {self.device}")

        # Setup environment
        self.env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
        self.preprocessor = AtariPreprocessor()

        # Load trained model
        self.state_shape = (4, 84, 84)
        self.n_actions = self.env.action_space.n
        self.model = DQN(self.state_shape, self.n_actions).to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Successfully loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using random agent.")
            self.model = None

    def get_action(self, state):
        """Get action from trained model (greedy policy)"""
        if self.model is None:
            return self.env.action_space.sample()  # Random action if no model

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.max(1)[1].item()

    def play_episode(self, render=True, save_video=False, video_path="gameplay.gif"):
        """Play a single episode and optionally save video"""
        state = self.preprocessor.reset(self.env)
        total_reward = 0
        steps = 0
        frames = []

        print("Starting episode...")

        while True:
            # Get action from model
            action = self.get_action(state)

            # Take action
            next_state, reward, done, info = self.preprocessor.step(self.env, action)
            total_reward += reward
            steps += 1

            # Capture frame for video
            if save_video or render:
                frame = self.env.render()
                frames.append(frame)

            # Show live gameplay (optional)
            if render and steps % 4 == 0:  # Show every 4th frame to reduce flicker
                cv2.imshow('Pac-Man DQN Agent', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            state = next_state

            if done:
                break

        if render:
            cv2.destroyAllWindows()

        # Save video if requested
        if save_video and frames:
            print(f"Saving video to {video_path}...")
            # Downsample frames for smaller file size
            imageio.mimsave(video_path, frames, fps=5)
            print(f"Video saved with {len(frames)} frames")

        return total_reward, steps, len(frames)

    def evaluate_agent(self, num_episodes=5, save_videos=False):
        """Evaluate agent over multiple episodes"""
        scores = []
        episode_lengths = []

        print(f"\nEvaluating agent over {num_episodes} episodes...")
        print("-" * 50)

        for episode in range(num_episodes):
            video_path = f"episode_{episode + 1}.gif" if save_videos else None
            score, steps, frames = self.play_episode(
                render=False,
                save_video=save_videos,
                video_path=video_path
            )

            scores.append(score)
            episode_lengths.append(steps)

            print(f"Episode {episode + 1:2d}: Score = {score:6.0f}, Steps = {steps:4d}")

        # Statistics
        print("-" * 50)
        print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"Best Score:    {np.max(scores):.0f}")
        print(f"Worst Score:   {np.min(scores):.0f}")
        print(f"Avg Steps:     {np.mean(episode_lengths):.1f}")

        return scores, episode_lengths

    def plot_q_values(self, state, action_names=None):
        """Visualize Q-values for current state"""
        if self.model is None:
            print("No model loaded - cannot show Q-values")
            return

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]

        if action_names is None:
            action_names = [f"Action {i}" for i in range(len(q_values))]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(q_values)), q_values)
        plt.title("Q-Values for Current State")
        plt.xlabel("Actions")
        plt.ylabel("Q-Value")
        plt.xticks(range(len(q_values)), action_names, rotation=45)

        # Highlight best action
        best_action = np.argmax(q_values)
        bars[best_action].set_color('red')

        plt.tight_layout()
        plt.show()

        print(f"Best action: {action_names[best_action]} (Q-value: {q_values[best_action]:.3f})")

    def interactive_demo(self):
        """Interactive demo with real-time Q-value display"""
        print("\nStarting interactive demo...")
        print("Press 'q' to quit, 's' to show Q-values, 'p' to pause")

        state = self.preprocessor.reset(self.env)
        total_reward = 0
        paused = False

        # Pac-Man action names (may vary by environment version)
        action_names = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
        action_names = action_names[:self.n_actions]  # Trim to actual number of actions

        while True:
            if not paused:
                action = self.get_action(state)
                next_state, reward, done, info = self.preprocessor.step(self.env, action)
                total_reward += reward

                print(
                    f"\rScore: {total_reward:6.0f} | Action: {action_names[action] if action < len(action_names) else action}",
                    end="")

                state = next_state

                if done:
                    print(f"\nGame Over! Final Score: {total_reward}")
                    break

            # Render frame
            frame = self.env.render()
            cv2.imshow('Pac-Man DQN Agent (Interactive)', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Handle keyboard input
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\nShowing Q-values...")
                self.plot_q_values(state, action_names)
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Test DQN Pac-Man Agent')
    parser.add_argument('--model', type=str, default='dqn_pacman_1000.pth',
                        help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to evaluate')
    parser.add_argument('--save-videos', action='store_true',
                        help='Save gameplay videos as GIFs')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo')
    parser.add_argument('--single-game', action='store_true',
                        help='Play single game with live rendering')

    args = parser.parse_args()

    # Create tester
    tester = GameplayTester(args.model)

    if args.interactive:
        tester.interactive_demo()
    elif args.single_game:
        print("Playing single game with live rendering...")
        score, steps, frames = tester.play_episode(render=True, save_video=args.save_videos)
        print(f"\nGame finished! Score: {score}, Steps: {steps}")
    else:
        # Standard evaluation
        scores, lengths = tester.evaluate_agent(args.episodes, args.save_videos)

        # Plot results
        if len(scores) > 1:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(scores, 'bo-')
            plt.title('Scores per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')

            plt.subplot(1, 2, 2)
            plt.hist(scores, bins=min(10, len(scores)), alpha=0.7)
            plt.title('Score Distribution')
            plt.xlabel('Score')
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()