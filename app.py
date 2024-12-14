# app.py

import streamlit as st
import gym
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from PIL import Image

# Load logo
st.set_page_config(page_title="Q-Learning Agent Dashboard", page_icon="🤖")
st.sidebar.image("logo.png", use_container_width=True)
st.title("Q-Learning Agent: CartPole-v1 Dashboard")

# Load environment and Q-table
env = gym.make("CartPole-v1", render_mode="human")
bins = 10  # Ensure consistent bins with training

# State bins for discretization
state_bins = [
    np.linspace(-4.8, 4.8, bins),  # Cart position
    np.linspace(-4, 4, bins),  # Cart velocity
    np.linspace(-0.418, 0.418, bins),  # Pole angle
    np.linspace(-4, 4, bins),  # Pole velocity
]

# Discretize state function
def discretize_state(state):
    return tuple([np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state))])

# Load Q-table
q_table_file = "Gym_q_table.pkl"
try:
    with open(q_table_file, "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    st.error("Trained Q-table not found. Train the agent and save the Q-table as 'Gym_q_table.pkl'.")
    st.stop()

# Interactive Controls
st.sidebar.header("Agent Settings")
epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", 0.01, 1.0, 0.01, 0.01)
render_env = st.sidebar.checkbox("Render Environment", value=False)
episodes = st.sidebar.number_input("Number of Episodes to Simulate", min_value=1, max_value=500, value=10)

# Training Simulation
if st.sidebar.button("Simulate"):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state)
        total_reward = 0
        done = False

        while not done:
            # Choose action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Step
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state)
            state = next_state
            total_reward += reward

            if render_env:
                env.render()

        rewards.append(total_reward)
        st.write(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    # Plot rewards
    fig, ax = plt.subplots()
    ax.plot(range(episodes), rewards, label="Rewards")
    ax.set_title("Reward vs Episodes")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    st.pyplot(fig)

# Real-time interaction
st.header("Test Agent in Real-Time")
test_button = st.button("Run Agent")

if test_button:
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    frames = []

    while not done:
        action = np.argmax(q_table[state])
        next_state, _, done, _, _ = env.step(action)
        state = discretize_state(next_state)
        env.render()

    st.success("Agent run completed.")
env.close()
