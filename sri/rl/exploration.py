import os
# Default to headless software rendering (Linux clusters); override by setting
# MUJOCO_GL yourself (e.g. MUJOCO_GL=glfw on macOS).
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import random
from PIL import Image
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['PPO', 'SAC', 'TQC'], default='PPO',
                        help='SB3 algorithm to use: PPO, SAC, TQC. Default is PPO.')
    parser.add_argument('--her', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to use HER. Default is True.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for the SB3 algorithm. Default is 0.99.')
    parser.add_argument('--learning-steps', type=int, default=10000000,
                        help='Number of learning steps. Default is 10000000.')
    # parser.add_argument('--test-steps', type=int, default=30000,
    #                     help='Number of steps to test the model. Default is 30000.')
    # parser.add_argument('--test-render-interval', type=int, default=100,
    #                     help='Number of steps between renders during testing. Default is 100.')
    parser.add_argument('--render-interval', type=int, default=10000,
                        help='Number of steps between renders during training. Default is 100.')
    parser.add_argument('--render-duration', type=int, default=-1,
                        help='Duration of each render during training in seconds. Default is entire episode.')
    parser.add_argument('--latent-dim', type=int, default=256, help='Number of neurons in each layer')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers in the network')
    parser.add_argument('--num-critics', type=int, default=2, help='Number of critics in TQC')
    parser.add_argument('--timestep', type=float, default=0.002,
                        help='Timestep for the simulation. Default is 0.002.')
    parser.add_argument('--target-update-interval', type=int, default=1,
                        help='Number of steps between target network updates. Default is 1.')
    parser.add_argument('--train-frequency', type=int, default=1,
                        help='Number of steps between gradient updates. Default is 1.')
    # parser.add_argument('--normalize-actions', action=argparse.BooleanOptionalAction, default=True,
    #                     help='Whether to normalize actions. Default is True.')
    # parser.add_argument('--record-observation-bounds', action=argparse.BooleanOptionalAction, default=False,
    #                     help='Whether to record empirical observation bounds. Default is False.')
    # parser.add_argument('--normalize-observations', action=argparse.BooleanOptionalAction, default=True,
    #                     help='Whether to normalize observations. Default is True.')
    # parser.add_argument('--diagnostic-interval', type=int, default=10000,
    #                     help='Number of steps between diagnostics. Default is 10000.')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of environments to run in parallel. Default is 1.')
    parser.add_argument('--save-freq', type=int, default=100000,
                        help='Number of steps between model saves. Default is 100000.')
    parser.add_argument('--weight-resets', default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to reset the policy after policy-reset-steps")
    parser.add_argument('--weight-reset-interval', default=10000, type=int,
                        help="Number of steps after which to reset the policy")
    parser.add_argument('--her-selection-strategy', default="future", type=str,
                        help="HER selection strategy. Default is future.")
    parser.add_argument('--learning-starts', default=10000, type=int,
                        help="Number of steps before training TQC/SAC. Default is 10000.")
    # parser.add_argument('--terminal-goals', default=False, action=argparse.BooleanOptionalAction,
    #                     help="Whether to terminate the episode when goals are reached. Default is False.")
    # parser.add_argument('--normalized-action-bounds', default=1.0, type=float,
    #                     help="Proportion of maximum/minimum action range represented by normalized 1/-1. \
    #                     Recommended is to scale up, e.g., 10 or 100, so actions are not too large. Default is 1.0.")
    # parser.add_argument('--action-scale', default=1.0, type=float,
    #                     help="Scale of actions. Can modify down for more fine-tuned control. Default is 1.0.")

pick_place_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-v2-goal-observable"]
# pick_place_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["pick-place-v2-goal-hidden"]
env = pick_place_cls(render_mode="rgb_array")
first_obs = env.reset()
# first, let's try just resetting and rendering a bunch of times
# to see if the goal changes
# for i in tqdm(range(100)):
    # obs = env.reset()
    # now let's see which parts of the observation have changed
    # we can do that with a simple subtraction
    # print(f"Difference between first observation and observation {i}: {obs[0] - first_obs[0]}")
    # img_array = env.render()
#     img = Image.fromarray(img_array, 'RGB')
#     # # let's save to a directory
#     os.makedirs("pick_place_goal_observable", exist_ok=True)
#     img.save("pick_place_goal_observable/env_state_{}.png".format(i))
#     # Okay, I couldn't tell from the images. Not even sure if it renders the goal
#     # Let's just get print the goal as an observation
#     # print(f"observation {i}: {obs[0]}")