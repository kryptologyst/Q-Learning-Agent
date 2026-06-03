import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import QLearningAgent, GridWorld


class TestQLearning:
    def test_train(self):
        env = GridWorld(5)
        agent = QLearningAgent(n_states=25, n_actions=4)
        history = agent.train(env, episodes=100)
        assert len(history) == 100
        assert history[-1] >= history[0]

    def test_q_table_shape(self):
        agent = QLearningAgent(n_states=25, n_actions=4)
        assert agent.q_table.shape == (25, 4)
