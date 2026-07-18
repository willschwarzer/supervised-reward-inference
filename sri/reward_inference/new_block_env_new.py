import random
import json
import math
import gym
from gym import spaces
from enum import IntEnum
import numpy as np
import torch
import torch.nn.functional as F

viewport_width = 5 # must be odd integer
large_number = 1000000000000
empty_block_multiplier = 3 # higher number = more empty blocks
on_block_multiplier = 20 # how much reward if on top of block
num_blocks = 3 # number of blocks which can be assigned a type
assert(viewport_width == int(viewport_width))
assert(viewport_width % 2 == 1)

viewport_iteration_arr = [i-viewport_width//2 for i in range(viewport_width)]

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NOOP = 4

class NewBlockEnv(gym.Env):

    def block_at(self, x, y):
        if (x,y) in self.eaten_blocks:
            return num_blocks

        state = random.getstate()
        random.seed(x * large_number + y)
        block = random.randint(0, num_blocks * empty_block_multiplier)

        if block < num_blocks:
            ret = block
        else:
            ret = num_blocks
        
        random.setstate(state)
        return ret

    def rendered_block_at(self, x, y):
        if (x,y) in self.eaten_blocks:
            return num_blocks

        state = random.getstate()
        random.seed(x * large_number + y)
        block = random.randint(0, num_blocks * empty_block_multiplier)
        
        if block < num_blocks:
            ret = block
        else:
            ret = num_blocks

        random.setstate(state)
        return ret

    def eat(self, x, y):
        self.eaten_blocks.add((x, y))

    def __init__(self, block_rewards, state_rep_model=None):
        super(NewBlockEnv, self).__init__()
        assert(len(block_rewards) == num_blocks)

        # if state_rep is not None:
        #     self.state_rep = state_rep
        # else:
        #     self.state_rep = self._get_dist_sums
        
        self.block_rewards = block_rewards
        self.state_rep_model = state_rep_model
        
        self.eaten_blocks = set()

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.MultiDiscrete(np.ones((viewport_width * viewport_width,), dtype=np.int64) * (num_blocks+1))
        

        self._init_obs()

    def _init_obs(self):
        self.nsteps = 0
        self.x = random.randint(0, large_number//2)
        self.y = random.randint(0, large_number//2)


    def _next_observation(self):
        res = np.zeros((viewport_width, viewport_width), dtype=np.int64)
        for ix, dx in enumerate(viewport_iteration_arr):
            for iy, dy in enumerate(viewport_iteration_arr):
                x = self.x + dx
                y = self.y + dy
                res[ix, iy] = self.block_at(x, y)
        return res

    def _next_render(self):
        res = np.zeros((viewport_width, viewport_width), dtype=np.int64)
        for ix, dx in enumerate(viewport_iteration_arr):
            for iy, dy in enumerate(viewport_iteration_arr):
                x = self.x + dx
                y = self.y + dy
                res[ix, iy] = self.rendered_block_at(x, y)
        return res

    # NOTE: This function modifies the state if the agent is standing on a block
    def _get_reward(self, alt_reward=[0, 0, 0]):
        obs = self._next_observation()
        if self.state_rep_model is not None:
            obs_tensor = torch.Tensor(obs).to(torch.int64).cuda() #(n_examples, L, S); might need to be float
            obs_tensor = F.one_hot(obs_tensor, num_classes=4).view(100)
            obs_tensor = obs_tensor.to(torch.float)
            rep = self.state_rep_model(obs_tensor).cpu().detach().numpy()
            reward = np.dot(rep, self.block_rewards)
        reward = 0
        alt_reward_val = 0
        for ix, dx in enumerate(viewport_iteration_arr):
            for iy, dy in enumerate(viewport_iteration_arr):
                manhattan_dist = abs(dx) + abs(dy)
                rendered_block_type = self.rendered_block_at(self.x + dx, self.y + dy)
                multiplier = on_block_multiplier if manhattan_dist == 0 else 1/manhattan_dist

                if manhattan_dist == 0:
                    # We are sitting on a block, eat it
                    self.eat(self.x, self.y)
                
                if rendered_block_type < num_blocks:
                    reward += self.block_rewards[rendered_block_type] * multiplier

                # if obs[ix, iy] == BlockType.POSITIVE:
                #     reward += self.block_power[rendered_block_type-1] * multiplier
                #     alt_reward_val += alt_reward[rendered_block_type-1] * multiplier
                # elif obs[ix, iy] == BlockType.NEGATIVE:
                #     reward -= self.block_power[rendered_block_type-1] * multiplier
                #     alt_reward_val += alt_reward[rendered_block_type-1] * multiplier
                # elif obs[ix, iy] == BlockType.NEUTRAL:
                #     alt_reward_val += alt_reward[rendered_block_type-1] * multiplier
        return reward, alt_reward_val
    
    def _take_action(self, action):
        self.nsteps += 1
        if action == Action.UP:
            self.y -= 1
        elif action == Action.DOWN:
            self.y += 1
        elif action == Action.LEFT:
            self.x -= 1
        elif action == Action.RIGHT:
            self.x += 1

    def step(self, action, alt_reward=[0,0,0], render=False):
        self._take_action(action)

        done = self.nsteps > 150

        if render:
            obs = self._next_render()
        else:
            obs = self._next_observation()

        reward, alt_reward_val = self._get_reward(alt_reward = alt_reward)

        if action != Action.NOOP:
            # Apply slight penalty for moving
            reward -= 0.005

        return obs.flatten(), reward, done, {'alt_reward': alt_reward_val}

    def reset(self):
        self._init_obs()
        return self._next_observation().flatten()

    def render(self, mode='human', close=False):
        if mode == 'rgb':
            return self._next_render()
        else:
            return self._next_observation()
