import stable_baselines3
import procgen
from stable_baselines3.common.evaluation import evaluate_policy
import gym

model_path = input('model_path > ')
env_name = input('env_name > ')
extra_info = input('extra_info > ')
num_tries = int(input('num_tries > '))

env = gym.make(env_name, extra_info=extra_info)

model = stable_baselines3.PPO.load(model_path, env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_tries)
print(mean_reward, std_reward)
