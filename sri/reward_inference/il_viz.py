import os
os.environ["XLA_FLAGS"]="--xla_gpu_force_compilation_parallelism=1"
import stable_baselines3
from stable_baselines3 import PPO
from object_env import ObjectEnv
from predict_transformer_nonlinear import NonLinearNet
import torch
import numpy as np
import wandb
from PIL import Image

def load_model(location, obs_size, horizon, rep_dim):
    demonstration_rep_dim = rep_dim
    state_rep_dim = rep_dim
    state_hidden_size = 2048
    num_state_layers = 2
    demonstration_hidden_size = 2048
    num_demonstration_layers = 2
    mlp = False
    demonstration_sigmoid = False
    ground_truth_phi = False
    
    net = NonLinearNet(demonstration_rep_dim, 
                       state_rep_dim, 
                       state_hidden_size, 
                       64, 
                       demonstration_hidden_size, 
                       obs_size, 
                       horizon, 
                       num_demonstration_layers, 
                       num_state_layers, 
                       mlp=mlp, 
                       demonstration_sigmoid=demonstration_sigmoid,
                       ground_truth_phi = ground_truth_phi)
    # net = net.cuda() # Maybe shouldn't make this cuda
    net.load_state_dict(torch.load(location))
    return net

def main():
    # load states, rewards, weights from "data/rings_nojax_multimove1000_{rewards,states,weights}.npy"
    states = np.load("data/rings_nojax_multimove1000_states.npy")
    # rewards = np.load("data/rings_nojax_multimove1000_rewards.npy")
    weights = np.load("data/rings_nojax_multimove1000_weights.npy")

    # convert to torch tensors
    # our models use float32, so we need to convert
    states = torch.from_numpy(states).float()
    weights = torch.from_numpy(weights).float()

    # load the reward inference model from "nshot_nojax.parameters"
    obs_size = 10 # 2*num_rings
    horizon = 50 # for rings env
    rep_dim = 10 # change this depending on the model
    net = load_model("nshot_nojax.parameters", obs_size, horizon, rep_dim)

    # # states are of shape (num_tasks, num_episodes, horizon, obs_size)
    # # we want to get the first trajectory/episode
    # demonstrations = states[0][:100]
    # # we actually don't need the rewards, since we can reconstruct them with the weights
    # weight = weights[0]
    # # now we need to get the demonstration_encoder and state_encoder from the model
    # demonstration_encoder = net.demonstration_encoder
    # state_encoder = net.state_encoder
    # # now we encode the demonstration
    # # dem_rep = demonstration_encoder(demonstration.unsqueeze(0)).squeeze()
    # dem_rep = demonstration_encoder(demonstrations.unsqueeze(0)).squeeze()

    # # now we need to make the env
    # # here's a sample call: ObjectEnv(weights, num_rings=num_rings, seed=seed, env_size=1, move_allowance=True, episode_len=50, min_block_dist=0.25, intersection=True, max_move_dist=0.1, block_thickness=2, single_move=single_move)
    # # we're doing multimove, num_rings=5, we don't need to set seed, we got weights (dem rep) from the trajectory
    # # also we set state_encoder = state_encoder, and ground_truth_object_rewards=weight
    # # (obviously splitting between multiple lines this time)
    # env = ObjectEnv(
    #     dem_rep,
    #     num_rings=5,
    #     env_size=1,
    #     move_allowance=True,
    #     episode_len=50,
    #     min_block_dist=0.25,
    #     intersection=True,
    #     max_move_dist=0.1,
    #     block_thickness=2,
    #     single_move=False,
    #     state_encoder=state_encoder,
    #     ground_truth_object_rewards=weight,
    # )
    
    # # now we can render the demonstration
    # ims = env.render_rollout(demonstrations[0], show_reward=True)

    # # ims is basically a video, so we can save it as a gif
    # # but it's only numpy right now, so we need to convert it to PIL images
    # # we can do this with the following code
    # # breakpoint()
    # # but first we need to reshape to be (num_frames, width, height, channel)
    # ims = ims.transpose(0, 2, 3, 1)
    # images = [Image.fromarray(im) for im in ims]
    # # now we can save it as a gif
    # # the duration is in ms, so 100ms = 10fps
    # # however, we want to save it in "runs/nshot_nojax/demonstrations/0.gif"
    # # first, obviously, make the directories if they don't exist
    # os.makedirs("runs/nshot_nojax/demonstrations", exist_ok=True)
    # images[0].save("runs/nshot_nojax/demonstrations/0.gif", save_all=True, append_images=images[1:], duration=1000, loop=0)

    for task_idx in range(5):
        demonstrations = states[task_idx][:100] # selecting the first 100 demonstrations for each task
        weight = weights[task_idx]
        demonstration_encoder = net.demonstration_encoder
        state_encoder = net.state_encoder
        dem_rep = demonstration_encoder(demonstrations.unsqueeze(0)).squeeze()

        # looping through the first five tasks to render them
        env = ObjectEnv(
            dem_rep,
            num_rings=5,
            env_size=1,
            move_allowance=True,
            episode_len=50,
            min_block_dist=0.25,
            intersection=True,
            max_move_dist=0.1,
            block_thickness=2,
            single_move=False,
            state_encoder=state_encoder,
            ground_truth_object_rewards=weight,
        )

        for render_task_idx in range(5):

            # render the demonstration for the first trajectory of each task using the demonstration encoding of the current task
            ims = env.render_rollout(states[render_task_idx][0], show_reward=True) 
            ims = ims.transpose(0, 2, 3, 1)
            images = [Image.fromarray(im) for im in ims]

            # creating directories for each task and render using the demonstration encoding of the current task
            dir_path = f"runs/nshot_nojax/demonstrations/task{task_idx}/render_task{render_task_idx}"
            os.makedirs(dir_path, exist_ok=True)
            images[0].save(f"{dir_path}/0.gif",
                           save_all=True, append_images=images[1:], duration=250, loop=0)

if __name__ == "__main__":
    main()