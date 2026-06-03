import numpy as np
from loguru import logger

def create_gridworld(size=5):
    from .model import GridWorld
    logger.info(f"GridWorld {size}x{size} created")
    return GridWorld(size)
