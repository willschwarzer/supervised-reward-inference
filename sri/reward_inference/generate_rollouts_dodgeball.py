print("NOTE run with python3.6 on base conda environment in october_miniconda_install")
import numpy as np

import time
import stable_baselines3
import gym
import procgen
import os
import os.path
#os.mkdir('models_dodgeball')
#from new_block_env import *

def num_to_str(x):
    if x == -1:
        return "negative"
    elif x == 0:
        return "neutral"
    elif x == 1:
        return "positive"

start_time = time.time()
examples = []
examples_per_type = 20
num_models = 81
first_time_save = True
exampleno = 0
for v1 in [-1, 0, 1]:
    for v2 in [-1, 0, 1]:
        for v3 in [-1, 0, 1]:
            for v4 in [-1, 0, 1]:
                listt = [v1, v2, v3, v4]
                st = "_".join([num_to_str(x) for x in listt])
                path = "models_dodgeball/" + st + '.zip'

                print("Loading", path)
                env = gym.make('procgen-dodgeball-v0', extra_info=(','.join([str(x) for x in listt])))
                model = stable_baselines3.PPO.load(path, env)

                print("Done")
                for _ in range(examples_per_type):
                    example_obs = []
                    obs = env.reset()
                    for i in range(250):
                        action, _states = model.predict(obs)
                        obs, rewards, dones, info = env.step(action)
                        example_obs.append(obs) #env.render(mode='rgb').flatten())
                    example = {"key": listt, "data": np.array(example_obs), "info": info}
                    examples.append(example)
                    if first_time_save:
                        first_time_save = False
                        print("first time save")
                        np.save("examples_dodgeball/examples_dodgeball_testsave.npy", examples)
                    current_time = time.time()
                    estimated_total_time = (current_time-start_time) * examples_per_type * num_models / len(examples)
                    time_remaining = start_time + estimated_total_time - time.time()
                    print("\rExamples:", len(examples), "elapsed:", int(100*(current_time-start_time))/100.0, "time remaining:", int(time_remaining/60*100)/100, "mins", end="")
                print("")
                #if len(examples) >= 5000:
                #    np.save("examples_dodgeball/examples_dodgeball_" + str(exampleno) + ".npy", examples)
                #    exampleno += 1
                #    examples = []

print("We did it")
#np.save("examples_dodgeball/examples_dodgeball_" + str(exampleno) + ".npy", examples)

#for npos in range(5):
#  for nneu in range(5-npos):
#    nneg = 4-npos-nneu
#
#    listt = []
#    st = ""
#    for i in range(npos):
#      listt.append(1)#BlockType.POSITIVE)
#      st += "positive_"
#    for i in range(nneu):
#      listt.append(0)#BlockType.NEUTRAL)
#      st += "neutral_"
#    for i in range(nneg):
#      listt.append(-1)#BlockType.NEGATIVE)
#      st += "negative_"
#    st = st[:-1]
