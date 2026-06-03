import numpy as np
from loguru import logger


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.reward_history_: list = []

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state: int, action: int, reward: float, next_state: int):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (reward + self.gamma * best_next - self.q_table[state, action])

    def train(self, env, episodes: int = 500) -> list:
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            self.reward_history_.append(total_reward)
            if (ep + 1) % 100 == 0:
                logger.info(f"Episode {ep+1}/{episodes}: reward={total_reward}")
        return self.reward_history_


class GridWorld:
    def __init__(self, size: int = 5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.state = (0, 0)

    def reset(self):
        self.state = (0, 0)
        return self._to_idx(self.state)

    def step(self, action):
        r, c = self.state
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(self.size - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.size - 1, c + 1)
        self.state = (r, c)
        done = self.state == self.goal
        reward = 10 if done else -1
        return self._to_idx(self.state), reward, done

    def _to_idx(self, pos):
        return pos[0] * self.size + pos[1]
