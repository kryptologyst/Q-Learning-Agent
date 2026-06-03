import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class QLVisualizer:
    @staticmethod
    def plot_rewards(history, save_path=None):
        plt.figure(figsize=(8, 5))
        plt.plot(history, alpha=0.4, color="steelblue", linewidth=1)
        window = max(1, len(history) // 20)
        smoothed = np.convolve(history, np.ones(window)/window, mode="valid")
        plt.plot(range(window-1, len(history)), smoothed, color="crimson", linewidth=2, label=f"MA({window})")
        plt.xlabel("Episode"); plt.ylabel("Total Reward")
        plt.title("Q-Learning Training"); plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
