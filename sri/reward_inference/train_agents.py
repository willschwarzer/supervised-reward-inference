print("NOTE run with python3.6 on base conda environment in october_miniconda_install")

import stable_baselines3
from new_block_env import *

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

        print("Training", st)
        env = NewBlockEnv(listt)
        model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)

        print("Saving...")
        model.save("models/" + st)
