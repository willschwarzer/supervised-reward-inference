print ("#########################################################OUTDATED DO NOT USE###############################################################")

import numpy as np
import stable_baselines3
import ray
import psutil
import argparse
import new_block_env 
import new_block_env_regression 
import os

@ray.remote
def get_examples(n_examples):
    examples = []
    for npos in range(4):
        for nneu in range(4-npos):
            nneg = 3-npos-nneu

            listt = []
            st = ""
            for i in range(npos):
                listt.append(new_block_env.BlockType.POSITIVE)
                st += "positive_"
            for i in range(nneu):
                listt.append(new_block_env.BlockType.NEUTRAL)
                st += "neutral_"
            for i in range(nneg):
                listt.append(new_block_env.BlockType.NEGATIVE)
                st += "negative_"
            st = st[:-1]

            print("Loading", st)
            env = new_block_env.NewBlockEnv(listt)
            model = stable_baselines3.PPO.load("models/" + st, env)

            print("Done")
            for _ in range(n_examples):
                example_obs = []
                obs = env.reset()
                for i in range(150):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, info = env.step(action)
                    example_obs.append(env.render(mode='rgb').flatten())
                example = {"key": listt, "data": np.array(example_obs), "rgb_decode": env.rgb_decode}
                examples.append(example)
                print("\rExamples:", len(examples), end="")
                if psutil.virtual_memory().percent > 95:
                    exit()
            print("")
    return examples

@ray.remote
def get_examples_regression(n_examples, non_linear=False):
    def string_to_block_type(st):
        st = st.lower()
        if st == 'negative':
            return new_block_env_regression.BlockType.NEGATIVE
        elif st == 'positive':
            return new_block_env_regression.BlockType.POSITIVE
        elif st == 'neutral':
            return new_block_env_regression.BlockType.NEUTRAL
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
        env = new_block_env_regression.NewBlockEnv(block_types, block_powers)
        model = stable_baselines3.PPO.load(models_directory + "/" + model_name, env)

        print("Done")
        for _ in range(n_examples):
            data = []
            obs = env.reset()
            for i in range(150):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                data.append((obs, rewards))
            if non_linear:
                example = {"block_types": block_types, "block_powers": block_powers, "data": data, "rgb_decode": env.rgb_decode}
            else:
                example = {"block_types": block_types, "block_powers": block_powers, "data": np.array(data), "rgb_decode": env.rgb_decode}
            examples.append(example)
            print("\rExamples:", len(examples), end="")
        print("")
    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regression', '-r', action='store_true')
    parser.add_argument('--non-linear', '-nl', action='store_true')
    parser.add_argument('--max-threads', '-mt', default=16, type=int)
    parser.add_argument('--out', '-o', type=str, default='examples')
    args = parser.parse_args()
    return args

all_examples = []
args = parse_args()
n_threads = min(args.max_threads, psutil.cpu_count()-1)
ray.init(num_cpus=n_threads)
# ray.init()
print(ray.available_resources())
print(psutil.cpu_count())
n_examples_per_thread = [10000//n_threads for i in range(n_threads-1)]
n_examples_per_thread.append(10000 - sum(n_examples_per_thread))
if args.regression:
    outs = ray.get([get_examples_regression.remote(n_examples, args.non_linear) for n_examples in n_examples_per_thread])
else:
    outs = ray.get([get_examples.remote(n_examples) for n_examples in n_examples_per_thread])

for out in outs:
    all_examples.extend(out)

np.save(args.out + '_' + str(len(all_examples)) + ".npy", all_examples)
