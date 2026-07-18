print("NOTE run with python3.6 on base conda environment in october_miniconda_install")

print("#######################################################################################OUT OF DATE DO NOT RUN###################################################################################")

import stable_baselines3
import random
import ray
import argparse
import psutil
from new_block_env_regression import *

def env_to_str(env):
    st = ""
    for i in range(3):
        if env.block_roles[i] == BlockType.POSITIVE:
            st += "positiveZ" + str(env.block_power[i]) + "_"
        elif env.block_roles[i] == BlockType.NEGATIVE:
            st += "negativeZ" + str(env.block_power[i]) + "_"
        elif env.block_roles[i] == BlockType.NEUTRAL:
            st += "neutralZ" + str(env.block_power[i]) + "_"
        else:
            raise RuntimeError("Unknown block type")
    return st[:-1]

@ray.remote
def train_agent(env):
    st = env_to_str(env)
    print("Training", st, "norm", sum([x ** 2 for x in env.block_power]))
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    print("Saving...")
    model.save("models_regression/" + st)
    print(st)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-threads', '-mt', default=16, type=int)
    args = parser.parse_args()
    return args
                
if __name__ == '__main__':
    args = parse_args()
    n_threads = min(args.max_threads, psutil.cpu_count()-1)
    ray.init(num_cpus=n_threads)
    # ray.init()
    print(ray.available_resources())
    print(psutil.cpu_count())
    envs = []
    for npos in range(4):
        for nneu in range(4-npos):
            nneg = 3-npos-nneu
            for _ in range(5):
                listt = []
                listt_powers = []
                for i in range(npos):
                    listt.append(BlockType.POSITIVE)
                    listt_powers.append(random.uniform(0, 1))
                for i in range(nneu):
                    listt.append(BlockType.NEUTRAL)
                    listt_powers.append(0.0)
                for i in range(nneg):
                    listt.append(BlockType.NEGATIVE)
                    listt_powers.append(random.uniform(0, 1))
                env = NewBlockEnv(listt, listt_powers)
                envs.append(env)
    ray.get([train_agent.remote(env) for env in envs])
    main()
