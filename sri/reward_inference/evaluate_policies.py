print("NOTE run with python3.6 on base conda environment in october_miniconda_install")

import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
from new_block_env import *

print("Model, mean reward, std reward")
for npos in range(4):
    for nneu in range(4-npos):
        nneg = 3-npos-nneu

        listt = []
        st = ""
        for i in range(npos):
            listt.append(BlockType.POSITIVE)
            st += "positive_"
        for i in range(nneu):
            listt.append(BlockType.NEUTRAL)
            st += "neutral_"
        for i in range(nneg):
            listt.append(BlockType.NEGATIVE)
            st += "negative_"
        st = st[:-1]

        env = NewBlockEnv(listt)
        model = stable_baselines3.PPO.load("models/" + st, env)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(st, mean_reward, std_reward)
