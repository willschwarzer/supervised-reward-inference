print("NOTE run with python3.6 on base conda environment in october_miniconda_install")

import procgen
import stable_baselines3
from new_block_env import *

def num_to_str(x):
    if x == -5:
        return "negative"
    elif x == 0:
        return "neutral"
    elif x == 1:
        return "positive"

for firstpos in [-5, 0, 1]:
    for secondpos in [-5, 0, 1]:
        for thirdpos in [-5, 0, 1]:
            listt = [firstpos, secondpos, thirdpos]
            st = "_".join([num_to_str(x) for x in listt])
            extra_info = ','.join([str(x) for x in listt])
            env = gym.make('procgen-miner-v0', extra_info=extra_info)

            print("Training", st)
            print("Extra info", extra_info)
            model = stable_baselines3.PPO("CnnPolicy", env, verbose=1)
            model.learn(total_timesteps=400000)

            print("Saving...")
            model.save("models_miner/" + st)
