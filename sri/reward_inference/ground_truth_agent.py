from new_block_env_regression import *
import stable_baselines3
import numpy as np

block_powers = [1/math.sqrt(3) for _ in range(3)]
block_types = [BlockType.POSITIVE] * 3
env = NewBlockEnv(block_types, block_powers)
agent = stable_baselines3.PPO("MlpPolicy", env, verbose=0)
agent.learn(total_timesteps=100000)
total_reward = 0
counter = 0
total_alt_reward = 0
n_reward_samples = 0
distance_to_arp = 0
for _ in range(100):
    obs = env.reset()
    action = agent.predict(obs)
    for i in range(150):
        obs, reward, done, info = env.step(action, alt_reward=block_powers)
        total_reward += reward
        total_alt_reward += info['alt_reward']
        n_reward_samples += 1
        action = agent.predict(obs)
    print(counter, total_reward/n_reward_samples, total_alt_reward/n_reward_samples, n_reward_samples, distance_to_arp)
