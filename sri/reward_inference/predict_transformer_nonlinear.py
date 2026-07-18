import os
# from utils import get_freest_gpu
# FREEST_GPU = get_freest_gpu()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(FREEST_GPU)
from new_block_env import *
from utils import generate_gadget_demonstration
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import itertools
from torch.utils.data import random_split, DataLoader
import numpy as np
import sys
from tqdm import tqdm
import wandb
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
import new_block_env_new
import ray
import psutil
from models import *

NUM_DEM_PER_REWARD = 100
NUM_AGENTS_PER_GPU = 8
rng = np.random.default_rng()

def parse_args():
    parser = argparse.ArgumentParser(description='Train supervised IRL models')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save-model', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether to save the model after every epoch')
    parser.add_argument('--save-to', type=str, default='nonlinear_model.parameters',
                        help='Destination for saved model')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--train-split', type=float, default=0.8, help='proportion of data to be in train split')
    parser.add_argument('--saved-model', type=str, default=None, help='location of saved model to load')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='sirl')
    parser.add_argument('--permute-types', action='store_true')
    parser.add_argument('--demonstration-sigmoid', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--env', default="metaworld", type=str)
    parser.add_argument('--num-demonstration-layers', default=1, type=int)
    parser.add_argument('--num-state-layers', default=1, type=int)
    parser.add_argument('--state-hidden-size', default=64, type=int)
    parser.add_argument('--demonstration-hidden-size', default=2048, type=int)
    parser.add_argument('--dem-encoder-type', type=str, default="transformer", help="transformer or lstm or set_transformer")
    parser.add_argument('--demonstration-rep-dim', default=3, type=int)
    parser.add_argument('--internal-tst-dim', default=100, type=int, 
                        help="dimension of the demonstration encoding before passing to set transformer")
    parser.add_argument('--state-rep-dim', default=3, type=int)
    parser.add_argument('--ground-truth-weights', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-shuffled', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--ground-truth-phi', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--rl-evaluation', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max-threads', default=50, type=int, help="For RL evaluation only")
    parser.add_argument('--num-rings', default=5, type=int, help="For ring env only")
    # parser.add_argument('--reward-batches', default=False, action=argparse.BooleanOptionalAction)
    # XXX what is reward batches for?
    parser.add_argument('--data-prefix', type=str, default="rings_multimove")
    parser.add_argument('--batch-by-task', default=True, action=argparse.BooleanOptionalAction, help="Do batching by task")
    parser.add_argument('--n', default=100, type=int, help="How many demonstrations to train on (if not batching by task, must be 1)")
    parser.add_argument('--rand-n', default=False, action=argparse.BooleanOptionalAction, help="Train on a random number of demonstrations within each task and epoch")
    parser.add_argument('--num-states', type=int, help="Number of iid states to predict rewards of per task (defaults to all)")
    parser.add_argument('--demonstration-type', type=str, default="normal",
                        help="normal or gadget")
    args = parser.parse_args()
    
    assert args.env == "rings", "remember to change the dataset names"
    assert not args.permute_types, "we're not sure this is correct, remember?"
    if args.ground_truth_phi or args.ground_truth_weights:
        # dem rep dim and state rep dim must be 10
        print("Using ground truth weights, so setting dem rep dim and state rep dim to 10")
        args.demonstration_rep_dim = 10
        args.state_rep_dim = 10
    return args

class NonlinearDataset(torch.utils.data.Dataset):
    def __init__(self, states, rewards, weights, num_classes=4):
        self.rewards = torch.Tensor(rewards).to(torch.float) #(n_examples, L)
        if num_classes == 0: # Indicates continuous environment, i.e. rings (at the moment)
            self.states = torch.Tensor(states).to(torch.float) #(n_examples, L, S);
        else: # Discrete environment
            self.states = torch.Tensor(states).to(torch.int64) #(n_examples, L, S); might need to be float
            self.states = F.one_hot(self.states, num_classes=num_classes).view(-1, 150, 25*num_classes)
        self.weights = weights # usually None
        if weights is not None:
            self.weights = torch.Tensor(weights).to(torch.float)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        if self.weights is not None:
            return self.states[index].cuda().to(torch.float), self.rewards[index].cuda(), self.weights[index].cuda() # returns a whole (L, S), (L, 1) demonstration
        else:
            return self.states[index].cuda().to(torch.float), self.rewards[index].cuda(), 0 # 0 here is in effect None, just not allowed to use None

def custom_collate_fn(batch):
    # useful for padding when we have variable length demonstrations
    max_n = max(len(item[0]) for item in batch)  # Assuming item[0] is 'return_dems'
    padded_batch = []
    
    for dems, states, rewards, weights in batch:
        padding_size = max_n - len(dems)
        padding_tensor = torch.zeros(padding_size, dems.shape[1], dems.shape[2]).cuda()
        padded_dems = torch.cat([dems, padding_tensor], dim=0)

        padded_batch.append((padded_dems, states, rewards, weights))

    # breakpoint()
    dems_batch, states_batch, rewards_batch, weights_batch = zip(*padded_batch)
    # states_batch, rewards_batch, weights_batch are all currently tuples of tensors
    # we need to convert them to tensors
    states_batch = torch.stack(states_batch)
    rewards_batch = torch.stack(rewards_batch)
    if type(weights_batch[0]) != int:
        weights_batch = torch.stack(weights_batch)
    dems_batch = torch.stack(dems_batch)
    # Convert states_batch, rewards_batch, weights_batch to tensors if necessary

    return dems_batch, states_batch, rewards_batch, weights_batch

# Dataset for data organized by task (i.e. reward)
# Allows for us to do n-shot, as well as using iid states separate from the
# demonstration trajectory for inference
# Currently only implemented for rings
class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, states, rewards, weights, n=1, rand_n=False, num_states=None, demonstration_type='normal', weights_only_for_gadget=True):
        self.rewards = torch.Tensor(rewards).to(torch.float) #(n_tasks, n_examples, L)
        self.states = torch.Tensor(states).to(torch.float) #(n_tasks, n_examples, L, S)
        self.weights = weights # usually None
        self.n = n
        self.rand_n = rand_n # if true, uses self.n as max n
        self.demonstration_type = demonstration_type.split(',')
        if num_states is None:
            self.num_states = self.states.shape[1]*self.states.shape[2] # e.g. 100*T
        else:
            self.num_states = num_states
        if weights is not None:
            self.weights = torch.Tensor(weights).to(torch.float)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        # Since this is a task dataset, index represents a reward function
        dems = self.states[index] # (n_examples, L, S)
        rewards = self.rewards[index] # (n_examples, L)
        if self.rand_n:
            n = rng.integers(0, self.n, endpoint=True) # zero-shot!
        else:
            n = self.n
        shuffled_dems = dems[torch.randperm(len(dems))]
        return_dems = shuffled_dems[:n]
        if 'gadget' in self.demonstration_type:
            gadget_demo = generate_gadget_demonstration(self.weights[index])  # Assuming the first weight is used for the gadget demo
            if 'normal' in self.demonstration_type:
                return_dems = torch.cat([gadget_demo.unsqueeze(0), return_dems], dim=0)
            else:
                return_dems = gadget_demo.unsqueeze(0)

        all_states = np.reshape(dems, (-1, dems.shape[-1]))
        all_rewards = np.reshape(rewards, (-1,))
        shuffled_idxs = rng.permutation(np.arange(len(all_states)))
        shuffled_states = all_states[shuffled_idxs]
        shuffled_rewards = all_rewards[shuffled_idxs]
        return_states = shuffled_states[:self.num_states]
        return_rewards = shuffled_rewards[:self.num_states]
        return_dems = return_dems.cuda()
        return_states = return_states.cuda()
        return_rewards = return_rewards.cuda()
        if self.weights is not None:
            return_weights = self.weights[index].cuda()
            return return_dems, return_states, return_rewards, return_weights
        else:
            return return_dems, return_states, return_rewards, 0 # 0 here is in effect None, just not allowed to use None
    
def get_splits(args):
    prefix = "data/"
    prefix += "shuffled_" if args.use_shuffled else ""
    state_path = prefix + args.data_prefix + "_observations.npy"
    reward_path = prefix + args.data_prefix + "_rewards.npy"
    weight_path = prefix + args.data_prefix + "_weights.npy"

    states = np.load(state_path, allow_pickle=True)
    rewards = np.load(reward_path, allow_pickle=True)
    if len(states.shape) == 3:
        # Data is long-form, i.e. not already grouped by task; group by task first
        assert args.num_examples % NUM_DEM_PER_REWARD == 0
        num_rewards = args.num_examples // NUM_DEM_PER_REWARD
        states_by_reward = np.reshape(states, (num_rewards, -1, states.shape[-2], states.shape[-1]))
        rewards_by_reward = np.reshape(rewards, (num_rewards, -1, rewards.shape[-1]))    
    else:
        states_by_reward = states
        rewards_by_reward = rewards
        num_rewards = len(states_by_reward)
    
    # Create train/val split by creating list of task indices, shuffling, and then indexing
    reward_idx_list = np.arange(num_rewards)
    rng.shuffle(reward_idx_list)
    train_length = int(num_rewards*args.train_split)
    train_idxs = reward_idx_list[:train_length]
    val_idxs = reward_idx_list[train_length:]
    train_states = states_by_reward[train_idxs]
    train_rewards = rewards_by_reward[train_idxs]
    val_states = states_by_reward[val_idxs]
    val_rewards = rewards_by_reward[val_idxs]
    
    if args.ground_truth_weights or args.rl_evaluation or "gadget" in args.demonstration_type:
        weights = np.load(weight_path, allow_pickle=True)
        weights_by_reward = np.reshape(weights, (num_rewards, -1, weights.shape[-1]))
        train_weights = weights_by_reward[train_idxs]
        train_weights_flattened = np.reshape(train_weights, (-1, train_weights.shape[-1]))
        val_weights = weights_by_reward[val_idxs]
        val_weights_flattened = np.reshape(val_weights, (-1, val_weights.shape[-1]))
    else:
        train_weights, val_weights, train_weights_flattened, val_weights_flattened = None, None, None, None    
            
    if not args.batch_by_task:
        # Flatten for final dataloader (can change this eventually if we want to train in pairs or more)
        train_states_flattened = np.reshape(train_states, (-1, train_states.shape[-2], train_states.shape[-1]))
        train_rewards_flattened = np.reshape(train_rewards, (-1, train_rewards.shape[-1]))
        val_states_flattened = np.reshape(val_states, (-1, val_states.shape[-2], val_states.shape[-1]))
        val_rewards_flattened = np.reshape(val_rewards, (-1, val_rewards.shape[-1]))
        train_dataset = NonlinearDataset(train_states_flattened, train_rewards_flattened, train_weights_flattened, num_classes)
        val_dataset = NonlinearDataset(val_states_flattened, val_rewards_flattened, val_weights_flattened, num_classes)
    else:
        weights_only_for_gadget = "gadget" in args.demonstration_type and not args.ground_truth_weights
        train_dataset = TaskDataset(
            train_states,
            train_rewards,
            train_weights,
            args.n,
            args.rand_n,
            args.num_states,
            demonstration_type=args.demonstration_type,
            weights_only_for_gadget=weights_only_for_gadget,
        )
        val_dataset = TaskDataset(
            val_states,
            val_rewards,
            val_weights,
            args.n,
            args.rand_n,
            args.num_states,
            demonstration_type=args.demonstration_type,
            weights_only_for_gadget=weights_only_for_gadget,
        )
    
    # training_data, validation_data = random_split(dataset, [train_length, val_length])
    # print("Num training", len(training_data))
    # print("Num validation", len(validation_data))
    if args.rand_n:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, validation_dataloader

@ray.remote(num_gpus=1/NUM_AGENTS_PER_GPU)
def evaluate_rl(dem_rep, state_encoder, weights):
    means = np.zeros((10, 3), dtype=float)
    for i in range(10):
        pred_env = new_block_env_new.NewBlockEnv(dem_rep, state_encoder)
        ground_truth_env = new_block_env_new.NewBlockEnv(weights)
        pred_model = stable_baselines3.PPO("MlpPolicy", pred_env, verbose=1)
        pred_model.learn(total_timesteps=100000)
        ground_truth_model = stable_baselines3.PPO("MlpPolicy", ground_truth_env, verbose=1)
        ground_truth_model.learn(total_timesteps=100000)
        random_model = stable_baselines3.PPO("MlpPolicy", ground_truth_env, verbose=1)
        mean_ground_truth_pred_model_ret = evaluate_policy(pred_model, ground_truth_env, n_eval_episodes=100)
        mean_ground_truth_ret = evaluate_policy(ground_truth_model, ground_truth_env, n_eval_episodes=100)
        mean_random_ret = evaluate_policy(random_model, ground_truth_env, n_eval_episodes=100)
        means[i] = [mean_ground_truth_pred_model_ret[0], mean_ground_truth_ret[0], mean_random_ret[0]]
    return np.mean(means, axis=0)

def get_batch_loss(net, loss_func, dem_batch, states_batch, rewards_batch, weights_batch, args, disp=False):
    if args.ground_truth_weights:
        prediction = net.forward(dem_batch, states_batch, weights_batch)
    else:
        prediction = net.forward(dem_batch, states_batch)
    loss = loss_func(prediction, rewards_batch)
    wandb.log({'train loss': loss})
    # loss.backward()
    if args.verbose and disp:
        print("Predicted vs actual rewards: \n", np.array(list(zip(prediction.cpu().detach().numpy(), rewards_batch.cpu().detach().numpy()))))
        print("Maximum error: \n", torch.max((prediction-rewards_batch)**2))
        worst = torch.argmax((prediction-rewards_batch)**2)
        print("Argmax of error: \n", prediction.flatten()[worst], rewards_batch.flatten()[worst])
        # print("demonstration reps: \n", net.demonstration_encoder(dem_batch))
    return loss

def train(args):
    wandb.init(project=args.wandb_project)
    wandb.config.update(args)
    
    train_dataloader, validation_dataloader = get_splits(args)
    
    # torch.cuda.set_device(f'cuda:{FREEST_GPU}')

    if "grid" in args.env:
        obs_size = 4*25
        horizon = 150
    elif "space" in args.env:
        obs_size = 6*25
        horizon = 150
    elif "ring" in args.env or "object" in args.env:
        obs_size = 2*args.num_rings
        horizon = 50
    else:
        raise NotImplementedError

    net = NonLinearNet(args.demonstration_rep_dim, 
                       args.state_rep_dim, 
                       args.internal_tst_dim,
                       args.state_hidden_size, 
                       64, 
                       args.demonstration_hidden_size, 
                       obs_size, 
                       horizon,
                       args.num_demonstration_layers, 
                       args.num_state_layers, 
                       mlp=args.mlp, 
                       demonstration_sigmoid=args.demonstration_sigmoid, 
                       dem_encoder_type=args.dem_encoder_type,
                       ground_truth_phi = args.ground_truth_phi).cuda()
    if args.saved_model:
        net.load_state_dict(torch.load(args.saved_model))

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()
    
    # assert args.rl_evaluation == args.ground_truth_weights, "You're using ground truth weights, are you sure you want to do that?"
    
    # TQDM inspiration from https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e

    
    for epoch in range(args.num_epochs):
        total_loss = 0
        n_samples = 0
        net.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for dem_batch, states_batch, rewards_batch, weights_batch in tepoch:
                
                optimizer.zero_grad()
                loss = get_batch_loss(net, loss_func, dem_batch, states_batch, rewards_batch, weights_batch, args, disp=tepoch.n%100==0)     
                loss.backward()
                optimizer.step()
                
                tepoch.set_description(f"Epoch {epoch}")
                total_loss += loss.item() * args.batch_size * horizon
                n_samples += args.batch_size * horizon
                tepoch.set_postfix({"avg loss": total_loss/n_samples})

        avg_loss = total_loss/(len(train_dataloader)*horizon*args.batch_size)
        #n_correct /= (3 * len(training))

        if args.save_model:
            # Base directory for all models
            base_model_dir = "models"

            # Subdirectory for this particular model
            model_subdir = os.path.join(base_model_dir, args.save_to)

            # Create the subdirectory if it doesn't exist
            if not os.path.exists(model_subdir):
                os.makedirs(model_subdir)

            # Construct the filename with wandb run name
            save_filename = os.path.join(model_subdir, f"{wandb.run.name}.parameters")

            # Save the model's state dictionary
            torch.save(net.state_dict(), save_filename)

            # Optional: Log the saved model file with wandb
            # wandb.save(save_filename)

            print(f"Model saved to {save_filename}")

        avg_training_loss = avg_loss
        print(f"Average training loss: {avg_training_loss}")
        wandb.log({'average train loss': avg_training_loss})
        total_loss = 0
        n_samples = 0
        net.eval()
        with torch.no_grad():
            with tqdm(validation_dataloader, unit="batch") as tepoch:
                for dem_batch, states_batch, rewards_batch, weights_batch in tepoch:
                    loss = get_batch_loss(net, loss_func, dem_batch, states_batch, rewards_batch, weights_batch, args, disp=tepoch.n%100==0) 
                    
                    total_loss += loss.item() * args.batch_size * horizon
                    n_samples += args.batch_size*horizon
                    tepoch.set_postfix({"avg validation loss": total_loss/n_samples})
        avg_loss = total_loss/(len(validation_dataloader)*horizon*args.batch_size)
        wandb.log({'average validation loss': avg_loss})
        
        


if __name__ == "__main__":
    args = parse_args()
    train(args)