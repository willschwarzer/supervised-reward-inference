import stable_baselines3
import procgen
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import sys

if len(sys.argv) == 6:
    _, model_path, env_name, extra_info, save_location, n_episodes = sys.argv
else:
    model_path = input('model_path > ')
    env_name = input('env_name > ')
    extra_info = input('extra_info > ')
    save_location = input('save_location > ')
    n_episodes = input('n_episodes > ')
n_episodes = int(n_episodes)

env = gym.make(env_name, extra_info=extra_info)

model = stable_baselines3.PPO.load(model_path, env)
obslist = []

for _ in range(n_episodes):
    obs = env.reset()
    obslist.append(obs)
    action, _ = model.predict(obs)
    for i in range(249):
        obs, _, _, _ = env.step(action)
        obslist.append(obs)
        action, _ = model.predict(obs)

np.save(save_location, np.array(obslist))
