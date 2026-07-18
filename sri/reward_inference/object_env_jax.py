import random
import json
import math
import gym
from gym import spaces
from enum import IntEnum
import numpy as np
import jax.numpy as jnp
from jax import random
from jax import jit
from functools import partial
import jax
import cv2 as cv
import palettable
from palettable.tableau import TableauLight_10, ColorBlind_10
import torch
# import torch.nn.functional as F

REWARD_BOUND = 100 # To bound reward
IM_SIZE = 256
BLOCK_SIZE = 8

class ObjectEnvJax(gym.Env):
    def __init__(self, object_rewards, state_encoder=None, seed=0, env_size=10, num_rings=3, move_allowance=True, im_size=256, block_size=16, block_thickness=4, episode_len=150, max_move_dist=1, act_space_bounds=1e9, min_block_dist=0.01, intersection=False, manual_clipping=False, single_move=False):
        super().__init__()
        if state_encoder is None: # Otherwise object_rewards is just the trajectory rep
            assert(len(object_rewards) == (num_rings*(num_rings-1)//2))
        
        assert not manual_clipping, "I really don't think you want to do that"
        assert move_allowance, "Always using this now"
        self.nsteps = 0
        self.object_rewards = object_rewards
        self.env_size = env_size
        self.num_rings = num_rings
        self.move_allowance = move_allowance
        self.im_size = im_size
        self.block_size = block_size
        self.block_thickness = block_thickness
        self.episode_len=episode_len
        self.max_move_dist = max_move_dist
        self.min_block_dist = min_block_dist
        self.intersection = intersection
        self.manual_clipping = manual_clipping
        self.single_move = single_move
        self.palette = TableauLight_10.colors
        self.state_encoder = state_encoder
        
        self.key = random.PRNGKey(seed)
        
        # Make a triangular-array-compatible weight vector to allow for efficient
        # computation later
        if state_encoder is None:
            tril = jnp.tril(jnp.ones([num_rings, num_rings]), k=-1) # Lower triangle of matrix, not including main diag
            self.expanded_object_rewards = tril.at[jnp.nonzero(tril)].set(object_rewards)
        
        self.state_bounds = jnp.array([-env_size/2, env_size/2], dtype=float)
        
        self.action_space = spaces.Box(low=-act_space_bounds, high=act_space_bounds, shape=(num_rings*2,))
        self.observation_space = spaces.Box(low=-env_size/2, high=env_size/2, shape=(num_rings, 2))
        
        self.state, self.key = _init_state(self.key, num_rings, env_size)
        

    def step(self, action, alt_reward=[0,0,0], render=False):
        action = jnp.array(action)
        if self.state_encoder is None:
            # breakpoint()
            self.state, reward = _step(self.state, action, self.expanded_object_rewards, self.state_bounds, self.move_allowance*self.max_move_dist, self.min_block_dist, self.intersection, self.manual_clipping, self.single_move)
            # self.state, reward = _step(self.state, action, self.expanded_object_rewards, self.state_bounds)
        else:
            self.state = _next_state(self.state, action, self.state_bounds, self.move_allowance, self.single_move)
            state_rep = self.state_encoder(torch.Tensor(np.asarray(jnp.reshape(self.state, (-1,)))))
            reward = np.dot(state_rep.detach().cpu().numpy(), self.object_rewards)

        self.nsteps += 1
        done = self.nsteps > self.episode_len

        return self.state, reward, done, {}

    def reset(self):
        # print(self.state)
        self.state, self.key = _init_state(self.key, self.num_rings, self.env_size)
        # self.state = jnp.clip(self.state, self.state_bounds[0] - 0.5, self.state_bounds[1] - 0.5)
        # self.state = self.state.at[1].set(self.state[0] + 0.5)
        # self.state = jnp.clip(self.state, self.state_bounds[0], self.state_bounds[1])
        self.nsteps = 0
        # print(_get_reward(self.state, self.expanded_object_rewards))
        
        return self.state

    def render(self, mode='human', close=False, circles=True, reward=None):
        im = np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)
        if circles:
            # size = int(self.min_block_dist*(self.im_size/self.env_size)/2)
            size = int(self.min_block_dist*(self.im_size/self.env_size)*2/5)
            margin = size + self.block_thickness
        else:
            margin = self.block_size//2 + self.block_thickness
        scale = (self.im_size-margin*2)/(self.env_size)
        pixel_centers = ((self.state + self.env_size/2)*scale + margin).astype(int)
        for idx, (x, y) in enumerate(pixel_centers):
            x, y = x.item(), y.item()
            if circles:
                cv.circle(im, (x, y), size, self.palette[idx], self.block_thickness)
            else:
                cv.rectangle(im, (x-self.block_size//2, y-self.block_size//2), (x+self.block_size//2, y+self.block_size//2), self.palette[idx], self.block_thickness)
        if reward is not None:
            (box_width, box_height), baseline = cv.getTextSize("{:.2f}".format(abs(reward)), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
            org = (0, box_height + baseline//2)
            c = int(abs(reward)*255)
            cv.putText(im, "{:.2f}".format(abs(reward)), org, cv.FONT_HERSHEY_SIMPLEX, 1, (255-c, 255, 255-c) if reward >= 0 else (255, 255-c, 255-c), 1, cv.LINE_AA)
        return np.moveaxis(im, -1, 0)
    
    def rollout_and_render(self, model, show_reward=False):
        self.reset()
        done = False
        ims = np.zeros((self.episode_len, 3, self.im_size, self.im_size), dtype=np.uint8)
        ims[0] = self.render()
        a, _ = model.predict(self.state)
        for i in range(1, self.episode_len):
            state, reward, _, _ = self.step(a)
            ims[i] = self.render(reward=(reward if show_reward else None))
            a, _ = model.predict(self.state)
        return ims
    
    def display_weights(self):
        size = 1024
        num_squares = self.num_rings
        square_width = 1024//num_squares
        im = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(self.num_rings-1):
            start = square_width*(i+1)
            cv.rectangle(im, (start, 0), (start+square_width, square_width), self.palette[i+1], -1)
            if i != self.num_rings - 1:
                cv.rectangle(im, (0, start), (square_width, start+square_width), self.palette[i], -1)
        for block_1 in range(self.num_rings):
            for block_2 in range(block_1+1, self.num_rings):
                start = (square_width*(block_2), square_width*(block_1+1))
                num = self.expanded_object_rewards[block_2][block_1]
                scale = 10/num_squares
                (box_width, box_height), baseline = cv.getTextSize("{:.2f}".format(abs(num)), cv.FONT_HERSHEY_SIMPLEX, scale, int(scale*1.5))
                org = (start[0] + (square_width-box_width)//2, start[1] + box_height + baseline + (square_width-box_height -baseline)//2)
                # scale = square_width*0.9/
                c = int(abs(num)*255)
                cv.putText(im, "{:.2f}".format(abs(num)), org, cv.FONT_HERSHEY_SIMPLEX, scale, (255-c, 255, 255-c) if num >= 0 else (255, 255-c, 255-c), int(scale*1.5), cv.LINE_AA)
                # cv.putText(im, "hello", (0, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 1, cv.LINE_AA)
        # cv.imwrite("test.png", im)
        return im
            

@partial(jit, static_argnums=(1, 2), backend='cpu')
def _init_state(key, num_rings, env_size):
    key, subkey = random.split(key)
    state = random.uniform(subkey, shape=(num_rings, 2), minval=-env_size/2, maxval=env_size/2)
    return state, key

@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def _step(state, action, weights, state_bounds, move_allowance, min_block_dist, intersection, manual_clipping, single_move):
    state = _next_state(state, action, state_bounds, move_allowance, single_move)
    reward = _get_reward(state, weights, min_block_dist, intersection)
    # if manual_clipping:
    #     reward = 0 # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #     violations = jnp.maximum(jnp.abs(action) - 1, 0)
    #     # reward = 10*jnp.any(violations) + 10*jnp.sum(violations)
    #     # reward *= -1
    #     # reward -= 10*(violations[-1] + violations[-2] + violations[-3] + violations[-4])
    #     reward -= 10*(violations[-1] + violations[-2])
    #     state *= 0
    #     # breakpoint()
    return state, reward

@partial(jit, static_argnums=(3, 4))
# @partial(jit, static_argnums=(3,))
def _next_state(state, action, state_bounds, move_allowance, single_move):
    # action = action.at[:-2].set(0)
    # action = action.at[-1].set(0)
    moves = jnp.reshape(action, (len(state), 2))
    # state = state.at[(-1, 1)].set(state[(-2, 1)])
    if move_allowance:
        # allowance = jnp.reshape(action[:len(state)], (len(state), 1))
        # moves = jnp.reshape(action[len(state):], (len(state), 2)) * allowance
        move_dists = jnp.sqrt(jnp.sum(moves**2, axis=-1))
        if single_move:
            mask = jnp.zeros(len(moves))
            mask = jnp.expand_dims(mask.at[jnp.argmax(move_dists)].set(1), 1)
            moves *= mask
            move_dists = jnp.sqrt(jnp.sum(moves**2, axis=-1))
        total_dist = jnp.sum(move_dists)
        moves *= jnp.minimum(move_allowance/total_dist, 1)
    state += moves
    state = jnp.clip(state, state_bounds[0], state_bounds[1])
    return state

@partial(jit, static_argnums=(2, 3))
def _get_reward(state, weights, min_block_dist, intersection):
    # state: (num_rings, 2)
    num_pairs = len(state)*(len(state)-1)//2
    obj_matrix_a = jnp.expand_dims(state, 1)
    obj_matrix_b = jnp.expand_dims(state, 0)
    squared_diffs = jnp.power(obj_matrix_a - obj_matrix_b, 2)
    squared_dists = jnp.sum(squared_diffs, axis=-1)
    dists = jnp.sqrt(squared_dists)
    if intersection:
        # arbitrary large distance for non-intersecting
        clipped_dists = jnp.where(dists <= min_block_dist, min_block_dist, 1000000)
    else:
        clipped_dists = jnp.clip(dists, a_min=min_block_dist)
    inv_dists = 1/clipped_dists
    reward = jnp.sum(inv_dists*weights)
    return reward