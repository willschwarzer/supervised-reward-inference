# from utils import get_freest_gpu
import os
# FREEST_GPU = get_freest_gpu()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(FREEST_GPU)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
from object_env import ObjectEnv
import numpy as np
import stable_baselines3
import imitation
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize
import psutil
import argparse
import os
from spaceinvaders.spaceinvaders_block_env_regression import *
from new_block_env_new import *
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import ray

NUM_WORKERS_PER_GPU = 20
rng = np.random.default_rng()

# @ray.remote(num_gpus=1/(NUM_WORKERS_PER_GPU))
@ray.remote
def get_examples_regression(model_dir, model_name, model_idx, num_examples, env, non_linear, verbose, chai_rollouts, num_rings, single_move, process_data):
    # stupidest thing I've ever seen, this doesn't load if it's not in the ray remote
    from imitation.data import rollout
    print("Loading model number", model_idx)
    def string_to_rewards(st):
        return np.array([float(num) for num in st.split("_")])

    weights = string_to_rewards(model_name)

    if verbose >= 1:
        print("Weights", weights)
        
    if "invaders" in env or "space" in env:
        env = SpaceInvadersBlockEnv(weights)
        L = 150
        state_size = 25
    elif "ring" in env or "object" in env:
        seed = rng.integers(10000)
        env = ObjectEnv(weights, num_rings=num_rings, seed=seed, env_size=1, move_allowance=True, episode_len=50, min_block_dist=0.25, intersection=True, max_move_dist=0.1, block_thickness=2, single_move=single_move)
        L = 50
        state_size = num_rings * 2
    elif "grid" in env:
        env = NewBlockEnv(weights)
        L = 150
        state_size = 25
    
    model = stable_baselines3.PPO.load(os.path.join(model_dir, model_name), env)

    if chai_rollouts:
        rollouts = rollout.rollout(
            model,
            DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=int(num_examples*1.3)),
            rng
        )
        for rollout in rollouts:
            if rollout.obs.shape[0] != L+2: # ask CHAI why this is L+2, not me
                rollouts.remove(rollout)
        assert len(rollouts) >= num_examples, "Too many bugged rollouts (L < 150)"
        rollouts = rollouts[:num_examples]
        return (rollouts, weights)
    else:
        if not process_data: # XXX copying code is bad, do as I say not as I do
            examples = []
            for _ in range(num_examples):
                data = []
                obs = env.reset()
                for i in range(L):
                    action, _states = model.predict(obs)
                    obs, reward, dones, _ = env.step(action)
                    data.append((np.asarray(obs), np.asarray(reward)))
                if non_linear:
                    example = {"data": data, "weights": weights}
                else:
                    assert False, "not ready yet"
                    # example = {"rewards": rewards, "data": data}
                    # example = {"block_types": block_types, "block_powers": block_powers, "data": np.array(data), "rgb_decode": env.rgb_decode}
                examples.append(example)
                if verbose >= 1:
                    print("\rExamples:", len(examples), end="")
            if verbose >= 1:
                print("")
            return examples
        else:
            trajectories = np.empty((num_examples, L, state_size), dtype=float)
            rewards = np.empty((num_examples, L), dtype=float)
            for ex_idx in range(num_examples):
                obs = env.reset()
                for t in range(L):
                    action, _states = model.predict(obs)
                    obs, reward, dones, _ = env.step(action)
                    trajectories[ex_idx][t] = obs.flatten()
                    rewards[ex_idx][t] = reward
                if verbose >= 1:
                    print("\rExample:", ex_idx, end="")
            if verbose >= 1:
                print("")
            return (trajectories, rewards, weights)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', '-md', type=str)
    parser.add_argument('--max-threads', '-mt', default=50, type=int)
    parser.add_argument('--out', '-o', type=str, default='examples')
    parser.add_argument('--non-linear', '-nl', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-examples', '-ne', type=int, default=1000)
    parser.add_argument('--num-models', '-nm', type=int, default=0)
    parser.add_argument('--env', type=str, default="rings")
    parser.add_argument('--verbose', '-v', type=int, default=0)
    parser.add_argument('--chai-rollouts', '-cr', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-rings', default=5, type=int, help="only used for ring env")
    parser.add_argument('--single-move', default=True, action=argparse.BooleanOptionalAction, help="only sued for ring env")
    parser.add_argument('--process-data', default=True, action=argparse.BooleanOptionalAction, help="if true, save as separate state, reward and weight files")
    args = parser.parse_args()
    return args

all_examples = []
args = parse_args()
n_threads = min(args.max_threads, psutil.cpu_count()-1)
ray.init(num_cpus=n_threads)
# ray.init()
print(ray.available_resources())
print(psutil.cpu_count())
models = os.listdir(args.model_dir)
print(f"Found {len(models)} agents")
print("Using agents starting from the end in the hope that the model didn't train on them")
num_models = len(models) if args.num_models == 0 else args.num_models
outs = ray.get([get_examples_regression.remote(args.model_dir, 
                                               model_name, 
                                               model_idx,
                                               args.num_examples, 
                                               args.env, 
                                               args.non_linear, 
                                               args.verbose, 
                                               args.chai_rollouts, 
                                               args.num_rings, 
                                               args.single_move, 
                                               args.process_data) for model_idx, model_name in enumerate(models[-num_models:])])

if args.chai_rollouts:
    rollouts = [ex[0] for ex in outs]
    weights = [ex[1] for ex in outs]
    for rollout in rollouts:
        all_examples.extend(rollout)
    serialize.save(f"data/{args.out}", all_examples)
    np.save(f"data/{args.out}_weights", weights)
elif not args.process_data:
    for out in outs:
        all_examples.extend(outs)
    np.save(args.out + '_' + str(len(all_examples)) + ".npy", all_examples)
else:
    state_out = np.array([out[0] for out in outs])
    reward_out = np.array([out[1] for out in outs])
    weight_out = np.array([out[2] for out in outs])
    np.save(f"data/{args.out}_states.npy", state_out)
    np.save(f"data/{args.out}_rewards.npy", reward_out)
    np.save(f"data/{args.out}_weights.npy", weight_out)