import random
import json
import math
import gym
from gym import spaces
from enum import IntEnum
import numpy as np

viewport_width = 5 # must be odd integer
large_number = 1000000000000
empty_block_multiplier = 3 # higher number = more empty blocks
on_block_multiplier = 20 # how much reward if on top of block
num_blocks = 3 # number of blocks which can be assigned a type
assert(viewport_width == int(viewport_width))
assert(viewport_width % 2 == 1)

viewport_iteration_arr = [i-viewport_width//2 for i in range(viewport_width)]

class BlockType(IntEnum):
    NEGATIVE = 0
    POSITIVE = 1
    NEUTRAL = 2
    EMPTY = 3

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NOOP = 4

class NewBlockEnv(gym.Env):

    def block_at(self, x, y):
        if (x,y) in self.eaten_blocks:
            return BlockType.EMPTY

        state = random.getstate()
        random.seed(x * large_number + y)
        block = random.randint(0, int(len(BlockType) * empty_block_multiplier))

        if block < num_blocks:
            ret = self.block_roles[block]
        else:
            ret = BlockType.EMPTY

        random.setstate(state)
        return ret

    def rendered_block_at(self, x, y):
        if (x,y) in self.eaten_blocks:
            return 0

        state = random.getstate()
        random.seed(x * large_number + y)
        block = random.randint(0, int(len(BlockType) * empty_block_multiplier))

        if block < num_blocks:
            block_color = self.block_roles[block]

            num_blocks_of_this_kind = 0
            block_type_locations = [0] * len(self.block_roles)
            for i, btype in enumerate(self.block_roles):
                if btype == block_color:
                    block_type_locations[num_blocks_of_this_kind] = i
                    num_blocks_of_this_kind += 1
            index_into_block_type = random.randint(0, num_blocks_of_this_kind-1)
            ret = 1 + self.color_permutations[block_type_locations[index_into_block_type]]
        else:
            ret = 0

        random.setstate(state)
        return ret

    def eat(self, x, y):
        self.eaten_blocks.add((x, y))

    def __init__(self, block_roles, block_power):
        super(NewBlockEnv, self).__init__()
        assert(len(block_roles) == num_blocks)
        assert(len(block_roles) == len(block_power))
        block_power_sum = 0
        for i in range(num_blocks):
            if block_roles[i] == BlockType.NEUTRAL:
                assert(block_power[i] == 0)
            block_power_sum += block_power[i] ** 2
        block_power_norm = math.sqrt(block_power_sum)
        if block_power_norm != 0:
            block_power = [x/block_power_norm for x in block_power]

        self.block_roles = block_roles
        self.block_power = block_power
        self.eaten_blocks = set()

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.MultiDiscrete(np.ones((viewport_width * viewport_width,), dtype=np.int64) * len(BlockType))

        self._init_obs()

    def _init_obs(self):
        self.nsteps = 0
        self.x = random.randint(0, large_number//2)
        self.y = random.randint(0, large_number//2)
        self.color_permutations = np.random.permutation(num_blocks)
        self.rgb_decode = np.empty(self.color_permutations.size+1, dtype=np.int32)
        self.rgb_decode[0] = BlockType.EMPTY
        for i in np.arange(self.color_permutations.size):
            self.rgb_decode[1+self.color_permutations[i]] = self.block_roles[i]


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
        reward = 0
        alt_reward_val = 0
        obs = self._next_observation()
        for ix, dx in enumerate(viewport_iteration_arr):
            for iy, dy in enumerate(viewport_iteration_arr):
                manhattan_dist = abs(dx) + abs(dy)
                rendered_block_type = self.rendered_block_at(self.x + dx, self.y + dy)
                multiplier = on_block_multiplier if manhattan_dist == 0 else 1/manhattan_dist

                if manhattan_dist == 0:
                    # We are sitting on a block, eat it
                    self.eat(self.x, self.y)

                if obs[ix, iy] == BlockType.POSITIVE:
                    reward += self.block_power[rendered_block_type-1] * multiplier
                    alt_reward_val += alt_reward[rendered_block_type-1] * multiplier
                elif obs[ix, iy] == BlockType.NEGATIVE:
                    reward -= self.block_power[rendered_block_type-1] * multiplier
                    alt_reward_val += alt_reward[rendered_block_type-1] * multiplier
                elif obs[ix, iy] == BlockType.NEUTRAL:
                    alt_reward_val += alt_reward[rendered_block_type-1] * multiplier
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
