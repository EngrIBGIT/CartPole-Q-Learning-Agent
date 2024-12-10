
# CartPole Q-Learning Agent

A simple reinforcement learning agent trained to balance a pole using Q-learning and OpenAI Gym.

## **Objective**
The aim of this project is to build a Q-learning agent that learns to interact with the CartPole-v1 environment in OpenAI Gym. The agent learns by optimizing its Q-table to maximize the total reward.

## **Prerequisites**
Before running the project, ensure you have the following installed:
- Python 3.7+
- OpenAI Gym (`pip install gym[classic_control]`)
- Numpy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- Seaborn (`pip install seaborn`)
- Pygame (`pip install pygame`)
- Plotly (`pip install plotly`)

To install all dependencies, run:
```bash
pip install -r requirements.txt

# Code Overview

## 1. Initialization
The `CartPole-v1` environment is initialized using OpenAI Gym. The state space and action space are defined:

```python
env = gym.make("CartPole-v1", render_mode="human")
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
````

**How the Agent Learns**

The agent learns through Q-learning, an off-policy reinforcement learning algorithm. It interacts with the environment, exploring various actions (e.g., moving left or right) and updating its Q-table based on the rewards it receives. 

The Q-value of a state-action pair is updated using the Bellman equation:

Q(s_t, a_t) ← Q(s_t, a_t) + α[R_t + γ * max_a' Q(s_t+1, a')] - Q(s_t, a_t)]

**Exploration vs. Exploitation:** The agent balances exploring random actions (with probability epsilon) and exploiting the learned Q-values to take the best possible action.

**Q-value Update:** The Q-value is updated using feedback from the environment, improving the agent's policy.

**Analysis & Insights**

* **Learning Efficiency:** Over time, as the agent explores the environment and updates its Q-table, the agent's performance improves, reflected in the increasing reward values.
* **Convergence:** As epsilon decays, the agent shifts from exploration to exploitation, converging to the optimal policy for balancing the pole.

**Interactive Visualization:** Plotly's interactive visualizations allow deeper analysis, such as inspecting rewards over time and understanding when the agent performs optimally.
