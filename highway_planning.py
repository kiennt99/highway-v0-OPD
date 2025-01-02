import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo

from deterministic import DeterministicPlannerAgent
from tqdm import tqdm

game_config = {
    "normalize_reward": True
}

env = gym.make("highway-v0", render_mode="rgb_array", config=game_config)
video_path = "./demo_videos"  # Specify the directory to save the video
env = RecordVideo(env, video_path, disable_logger=True, step_trigger = lambda x: x == 1)
env.unwrapped.config["simulation_frequency"] = 30

(obs, info), done = env.reset(), False

# Make agent
agent_config = {
    "__class__": "<class 'deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method": "simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = DeterministicPlannerAgent(env, agent_config)

# Initialize tracking
cumulative_reward = 0
rewards = []
steps = []

# Run episode
for step in tqdm(range(env.unwrapped.config["duration"] or 30), desc="Running...", leave=True):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    # Update cumulative reward
    cumulative_reward += reward
    rewards.append(cumulative_reward)
    steps.append(step)
    
    # Render environment (optional)
    # env.render()
    
    if done:
        break

env.close()

# Plot the cumulative rewards
plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, marker='o')
plt.title(f"Cumulative Reward Over Steps (budget: {agent_config["budget"]})")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.show()
