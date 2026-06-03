import typer
import sys
from loguru import logger

from .config import settings
from .model import QLearningAgent, GridWorld
from .visualizer import QLVisualizer

app = typer.Typer(help="Q-Learning Agent CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def train(episodes: int = typer.Option(500, help="Training episodes")):
    logger.info(f"Training Q-Learning agent ({episodes} episodes)...")
    env = GridWorld(5)
    agent = QLearningAgent(n_states=25, n_actions=4)
    history = agent.train(env, episodes)
    logger.info(f"Final avg reward (last 50): {np.mean(history[-50:]):.1f}")
    QLVisualizer.plot_rewards(history, save_path=settings.plots_dir / "qlearning_rewards.png")
    logger.success("Done!")


if __name__ == "__main__":
    app()
