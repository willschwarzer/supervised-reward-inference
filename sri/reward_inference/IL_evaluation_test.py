import os
import sys
import time
import shutil
from utils import get_freest_gpu, convert_chai_rollouts
FREEST_GPU = get_freest_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FREEST_GPU)
import argparse
import numpy as np
from models import NonLinearNet
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import load
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.bc import BC
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import torch
import torch.nn.functional as F
from new_block_env_new import NewBlockEnv
from object_env import ObjectEnv
import ray
import wandb
from tqdm import tqdm

NUM_AGENTS_PER_GPU = 10
MAX_MODEL_BATCH = 256
rng = np.random.default_rng()

def parse_args():
    parser = argparse.ArgumentParser(description='Train supervised IRL models')
    parser.add_argument('--data', type=str, default='grid_regression_with_weights_chai_100000',
                        help='location of rollouts')
    parser.add_argument('--weights', type=str, default='weights_chai_100000.npy')
    parser.add_argument('--model', '-sm', type=str, default='ground_truth_phi_test.parameters', help='location of saved model to load')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--wandb-project', '-wp', type=str, default='sirl_evaluation')
    parser.add_argument('--max-threads', '-mt', default=10, type=int)
    parser.add_argument('--average-traj-reps', '-atr', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--env', default='rings', type=str)
    parser.add_argument('--agent-directory', default='ring_agents', type=str)
    parser.add_argument('--num-rings', default=5, type=int, help="Only for ring env")
    parser.add_argument('--single-move', action=argparse.BooleanOptionalAction, help="Only for ring env")
    parser.add_argument('--num-rollouts-per-agent', default=10000, type=int)
    parser.add_argument('--rl-its', default=400000, type=int, help="Number of iterations to run reinforcement learning")
    parser.add_argument('--adv-its', default=1000000, type=int, help="Number of iterations to run adversarial algs")
    parser.add_argument('--bc-epochs', default=100, type=int)
    parser.add_argument('--num-trials', default=10, type=int)
    parser.add_argument('--num-eval-episodes', default=100, type=int, help="Number of episodes to evaluate on")
    parser.add_argument('--num-random-agents', default=10, type=int)
    parser.add_argument('--percentile-evaluation', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--agent-save-dir', default="imitation_agents", type=str)
    parser.add_argument('--save-agents', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--overwrite-agents', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--save-dir', default="weights_and_reps", type=str)
    args = parser.parse_args()
    return args

# @ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
@ray.remote
def evaluate_random(env, num_eval_episodes):
    agent = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    mean_random_ret, _ = evaluate_policy(agent, env, n_eval_episodes=num_eval_episodes)
    return mean_random_ret, agent

# @ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
@ray.remote
def evaluate_random_rew(env, agent_name, num_eval_episodes, save_dest):
    agent = stable_baselines3.PPO.load(agent_name, env)
    mean_random_ret, _ = evaluate_policy(agent, env, n_eval_episodes=num_eval_episodes)
    agent.save(os.path.join(save_dest, "random_rew", str(mean_random_ret)))
    return mean_random_ret

# @ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
@ray.remote
def evaluate_ground_truth(env, rl_its, num_eval_episodes, save_dest):
    agent = stable_baselines3.PPO("MlpPolicy", env, verbose=1, device='cpu')
    agent.learn(total_timesteps=rl_its)
    mean_ground_truth_ret, _ = evaluate_policy(agent, env, n_eval_episodes=num_eval_episodes)
    agent.save(os.path.join(save_dest, "ground_truth", str(mean_ground_truth_ret)))
    return mean_ground_truth_ret

# @ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
@ray.remote
def evaluate_sirl(pred_env, ground_truth_env, rl_its, num_eval_episodes, save_dest):
    pred_agent = stable_baselines3.PPO("MlpPolicy", pred_env, verbose=1, device='cpu')
    pred_agent.learn(total_timesteps=rl_its)
    mean_ground_truth_pred_agent_ret, _ = evaluate_policy(pred_agent, ground_truth_env, n_eval_episodes=num_eval_episodes)
    # Save agent
    pred_agent.save(os.path.join(save_dest, "sirl", str(mean_ground_truth_pred_agent_ret)))
    # Also save the environment's predicted reward weights
    np.save(os.path.join(save_dest, "sirl", str(mean_ground_truth_pred_agent_ret) + "_weights"), pred_env.object_rewards.detach().numpy())
    return mean_ground_truth_pred_agent_ret

# @ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
@ray.remote
def evaluate_adversarial(rollouts, env, alg, adv_its, num_eval_episodes, save_dest):
    trainer = GAIL if alg == 'gail' else AIRL
    
    # venv = DummyVecEnv([lambda: env] * 8)
    venv = DummyVecEnv([lambda: env])
    
    learner = stable_baselines3.PPO(
        env=venv,
        policy='MlpPolicy',
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        device='cpu'
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )
    adversarial_trainer = trainer(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )
    adversarial_trainer.train(adv_its)
    mean_ret, _ = evaluate_policy(learner, venv, n_eval_episodes=num_eval_episodes)
    learner.save(os.path.join(save_dest, alg, str(mean_ret)))
    return mean_ret

@ray.remote
def evaluate_bc(rollouts, env, bc_epochs, num_eval_episodes, save_dest):
    
    bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
    demonstrations=rollouts,
    )
    
    bc_trainer.train(n_epochs=bc_epochs)
    mean_ret, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=num_eval_episodes)
    bc_trainer.policy.save(os.path.join(save_dest, 'bc', str(mean_ret)))
    return mean_ret
    
def load_model(location, obs_size, horizon):
    trajectory_rep_dim = 100
    state_rep_dim = 100
    state_hidden_size = 2048
    num_state_layers = 2
    trajectory_hidden_size = 2048
    num_trajectory_layers = 2
    mlp = False
    trajectory_sigmoid = False
    lstm = False
    ground_truth_phi = False
    
    net = NonLinearNet(trajectory_rep_dim, 
                       state_rep_dim, 
                       state_hidden_size, 
                       64, 
                       trajectory_hidden_size, 
                       obs_size, 
                       horizon, 
                       num_trajectory_layers, 
                       num_state_layers, 
                       mlp=mlp, 
                       trajectory_sigmoid=trajectory_sigmoid, 
                       lstm=lstm, 
                       ground_truth_phi = ground_truth_phi)
    # net = net.cuda() # Maybe shouldn't make this cuda
    net.load_state_dict(torch.load(location))
    return net

def main(args):
    obs_size, horizon, dtype = get_env_specs(args.env, args.num_rings)
    model = load_model(args.model, obs_size, horizon)
    state_encoder = model.state_encoder
    trajectory_encoder = model.trajectory_encoder.cuda()
    weights_data = np.load(f"data/{args.weights}")
    rollouts = load(f"data/{args.data}")
    states, _ = convert_chai_rollouts(rollouts, horizon, obs_size, dtype)
    state_batches = torch.Tensor(states.reshape(-1, args.num_rollouts_per_agent, horizon, obs_size))

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    weights_list = []
    traj_reps_list = []

    for i, (weights, traj_batch) in tqdm(enumerate(zip(weights_data, state_batches))):
        sirl_in = traj_batch[:, :, :].unsqueeze(0).cuda()
        traj_rep = trajectory_encoder(sirl_in).squeeze().detach().cpu().numpy()
        traj_reps_list.append(traj_rep)
        weights_list.append(weights)

    np.save(os.path.join(save_dir, "weights.npy"), np.array(weights_list))
    np.save(os.path.join(save_dir, "traj_reps.npy"), np.array(traj_reps_list))

def get_env_specs(env_name, num_rings=None):
    if "grid" in env_name:
        obs_size = 4 * 25
        horizon = 150
        dtype = int
    elif "space" in env_name:
        obs_size = 6 * 25
        horizon = 150
        dtype = int
    elif "ring" in env_name or "object" in env_name:
        obs_size = 2 * num_rings
        horizon = 50
        dtype = float
    else:
        raise NotImplementedError
    return obs_size, horizon, dtype


if __name__ == "__main__":
    args = parse_args()
    main(args)