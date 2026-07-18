print("NOTE run with python3.6 on base conda environment in october_miniconda_install")

import numpy as np
import stable_baselines3
from new_block_env_regression import *
import os

def string_to_block_type(st):
    st = st.lower()
    if st == 'negative':
        return BlockType.NEGATIVE
    elif st == 'positive':
        return BlockType.POSITIVE
    elif st == 'neutral':
        return BlockType.NEUTRAL
    raise RuntimeError("Unknown block type")

models_directory = "models_regression"
models = os.listdir(models_directory)
examples = []
print(f"Found {len(models)} models")
for model_name in models:
    blocks = [{'type': x[0], 'power': float(x[1])} for x in [y.split("Z") for y in model_name.split(".zip")[0].split("_")]]

    block_types = [string_to_block_type(x['type']) for x in blocks]
    block_powers = [x['power'] for x in blocks]

    print("Loading", model_name)
    print("Types", block_types)
    print("Powers", block_powers)
    env = NewBlockEnv(block_types, block_powers)
    model = stable_baselines3.PPO.load(models_directory + "/" + model_name, env)

    print("Done")
    for _ in range(10000):
        example_obs = []
        obs = env.reset()
        for i in range(150):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            example_obs.append(env.render(mode='rgb').flatten())
        example = {"block_types": block_types, "block_powers": block_powers, "data": np.array(example_obs), "rgb_decode": env.rgb_decode}
        examples.append(example)
        print("\rExamples:", len(examples), end="")
    print("")

np.save("examples_" + str(len(examples)) + "_regression.npy", examples)
