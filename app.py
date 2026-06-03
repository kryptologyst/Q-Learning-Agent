import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.model import QLearningAgent, GridWorld

st.set_page_config(page_title="Q-Learning", page_icon="🧠", layout="wide")
st.title("🧠 Q-Learning Agent")
st.markdown("Tabular Q-Learning on a 5x5 GridWorld — navigate from (0,0) to (4,4).")

c1, c2, c3 = st.columns(3)
with c1:
    episodes = st.slider("Episodes", 100, 2000, 500, 100)
with c2:
    alpha = st.slider("Alpha", 0.01, 1.0, 0.1, 0.01)
with c3:
    epsilon = st.slider("Epsilon", 0.0, 0.5, 0.1, 0.01)

if st.button("Train Agent", type="primary"):
    with st.spinner(f"Training for {episodes} episodes..."):
        env = GridWorld(5)
        agent = QLearningAgent(n_states=25, n_actions=4, alpha=alpha, epsilon=epsilon)
        history = agent.train(env, episodes)
    st.success(f"Final avg reward (last 50): **{np.mean(history[-50:]):.1f}**")
    df = pd.DataFrame({"Episode": range(len(history)), "Reward": history})
    df["Smoothed"] = df["Reward"].rolling(max(1, len(history)//20)).mean()
    st.line_chart(df.set_index("Episode")[["Reward", "Smoothed"]])
