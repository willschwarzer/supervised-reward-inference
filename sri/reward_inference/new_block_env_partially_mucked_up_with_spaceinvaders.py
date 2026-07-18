import random
import json
import gym
from gym import spaces
from enum import IntEnum
import numpy as np

viewport_width = 5 # must be integer
large_number = 1000000000000
empty_block_multiplier = 3 # higher number = more empty blocks
on_block_multiplier = 20 # how much reward if on top of block
num_blocks = 3 # number of blocks which can be assigned a type
assert(viewport_width == int(viewport_width))

#viewport_iteration_arr = [i-viewport_width//2 for i in range(viewport_width)]

class BlockType(IntEnum):
    NEGATIVE = 0
    POSITIVE = 1
    NEUTRAL = 2
    PLAYER = 3
    BULLET = 4
    EMPTY = 5

class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    FIRE = 2
    NOOP = 3

class SpaceInvadersBlockEnv(gym.Env):

    #def block_at(self, x, y):
    #    if (x,y) in self.eaten_blocks:
    #        return BlockType.EMPTY

    #    state = random.getstate()
    #    random.seed(x * large_number + y)
    #    block = random.randint(0, int(len(BlockType) * empty_block_multiplier))

    #    if block < num_blocks:
    #        ret = self.block_roles[block]
    #    else:
    #        ret = BlockType.EMPTY

    #    random.setstate(state)
    #    return ret

    #def rendered_block_at(self, x, y):
    #    if (x,y) in self.eaten_blocks:
    #        return 0

    #    state = random.getstate()
    #    random.seed(x * large_number + y)
    #    block = random.randint(0, int(len(BlockType) * empty_block_multiplier))

    #    if block < num_blocks:
    #        block_color = self.block_roles[block]

    #        num_blocks_of_this_kind = 0
    #        block_type_locations = [0] * len(self.block_roles)
    #        for i, btype in enumerate(self.block_roles):
    #            if btype == block_color:
    #                block_type_locations[num_blocks_of_this_kind] = i
    #                num_blocks_of_this_kind += 1
    #        index_into_block_type = random.randint(0, num_blocks_of_this_kind-1)
    #        ret = 1 + self.color_permutations[block_type_locations[index_into_block_type]]
    #    else:
    #        ret = 0

    #    random.setstate(state)
    #    return ret

    #def eat(self, x, y):
    #    self.eaten_blocks.add((x, y))

    def __init__(self, block_roles):
        super(NewBlockEnv, self).__init__()
        assert(len(block_roles) == num_blocks)

        self.block_roles = block_roles
        self.eaten_blocks = set()

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.MultiDiscrete(np.ones((viewport_width * viewport_width,), dtype=np.int64) * len(BlockType))

        self._init_obs()

    def _init_obs(self):
        self.nsteps = 0
        self.initial_seed = random.randint(0, large_number//2)
        self.color_permutations = np.random.permutation(num_blocks)

        # ships is an array of numbers of size num_blocks+2
        # it contains the "color values" of each block,
        # which means 0 = empty, 1 = player, 2 = a bullet, 3+ = a block type
        # use it to index into self.rgb_decode to get block type
        # ships should NEVER contain any 1 or 2 values

        self.ships = np.zeros(viewport_width * (viewport_width-1), dtype=np.int32)
        self.position = viewport_width // 2
        self.rgb_decode = np.empty(self.color_permutations.size+2, dtype=np.int32)
        self.rgb_decode[0] = BlockType.EMPTY
        self.rgb_decode[1] = BlockType.PLAYER
        self.rgb_decode[2] = BlockType.BULLET
        self.bullets = [] # each bullet is tuple of (y, x, marked_for_deletion_due_to_hitting_target)
        for i in np.arange(self.color_permutations.size):
            self.rgb_decode[3+self.color_permutations[i]] = self.block_roles[i]


    def _next_observation(self):
        assert(1 not in self.ships)
        assert(2 not in self.ships)
        res = np.zeros((viewport_width, viewport_width), dtype=np.int64)
        i = 0
        for y in range(viewport_width-1):
            for x in range(viewport_width):
                res[y, x] = self.rgb_decode[self.ships[i]]
                if res[y, x] == BlockType.PLAYER:
                    raise RuntimeError("uh oh")
                i += 1
        for x in range(viewport_width):
            if x == self.position:
                res[y, x] = BlockType.PLAYER
            else:
                res[y, x] = BlockType.EMPTY
        for bullet in self.bullets:
            y, x, _ = bullet
            res[y, x] = BlockType.BULLET
        return res

    # Note to self:
    # What should be shown in obs?
    # What in render?
    # How are the colors done?

    def _next_render(self):
        assert(1 not in self.ships)
        assert(2 not in self.ships)
        res = np.zeros((viewport_width, viewport_width), dtype=np.int64)
        i = 0
        for y in range(viewport_width-1):
            for x in range(viewport_width):
                res[y, x] = self.ships[i]
                i += 1
        for x in range(viewport_width):
            if x == self.position:
                res[y, x] = 1 # ship
            else:
                res[y, x] = 0 # empty
        for bullet in self.bullets:
            y, x, _ = bullet
            res[y, x] = 2 # bullet
        return res

    # NOTE: This function was usurped by _take_action, ignore below note
    # OLD NOTE: This function modifies the state if the agent is shooting
    def _get_reward(self):
        pass
        """
        reward = 0
        obs = self._next_observation()
        for ix, dx in enumerate(viewport_iteration_arr):
            for iy, dy in enumerate(viewport_iteration_arr):
                manhattan_dist = abs(dx) + abs(dy)
                multiplier = on_block_multiplier if manhattan_dist == 0 else 1/manhattan_dist

                if manhattan_dist == 0:
                    # We are sitting on a block, eat it
                    self.eat(self.x, self.y)

                if obs[ix, iy] == BlockType.POSITIVE:
                    reward += multiplier
                elif obs[ix, iy] == BlockType.NEGATIVE:
                    reward -= multiplier
        return reward
        """

    def _generate_new_ship(self):
        val = random.randint(0, int(num_blocks * empty_block_multiplier))
        if val < num_blocks:
            return 3 + val # read self.ships info for why this is three
        else:
            return 0 # empty

    def _take_action(self, action):
        # Increment steps
        self.nsteps += 1

        # Begin reward calculation
        reward = 0

        # Bring ships forward
        self.ships = np.roll(self.ships, 1)
        if self.ships[0] != 0:
            # Nonempty ship block got through!
            if self.ships[0] == 1 or self.ships[0] == 2:
                raise RuntimeError("Somehow a player or bullet block got into ships?")
            elif self.ships[0] >= 3:
                ship_block_type = self.rgb_decode[self.ships[0]]
                if ship_block_type == BlockType.POSITIVE:
                    reward += 1
                elif ship_block_type == BlockType.NEGATIVE:
                    reward -= 1
                elif ship_block_type == BlockType.NEUTRAL:
                    pass
                else:
                    raise RuntimeError("Something's very wrong part 2.")
            else:
                raise RuntimeError("Something's very wrong.")
        self.ships[0] = self._generate_new_ship()

        # Take the action, add new bullet if necessary
        if action == Action.LEFT:
            self.position = max(0, self.position-1)
        elif action == Action.RIGHT:
            self.position = min(viewport_width-1, self.position+1)
        elif action == Action.FIRE:
            bullets.append((viewport_width-1, self.position, False))
        elif action == Action.NOOP:
            pass

        # Update all bullets
        new_bullets = []
        for bullet in self.bullets:
            y, x, marked_for_deletion = bullet
            if not marked_for_deletion and y > 0:
                y -= 1
                ships_index = y * viewport_width + x
                assert(ships_index == int(ships_index))
                assert(ships_index < viewport_width)
                assert(ships_index >= 0)
                if self.ships[ships_index] != 0: #non-empty, we've hit a ship
                    ship_block_type = self.rgb_decode[self.ships[ships_index]]
                    if ship_block_type == BlockType.POSITIVE:
                        reward -= 1
                    elif ship_block_type == BlockType.NEGATIVE:
                        reward += 1
                    elif ship_block_type == BlockType.NEUTRAL:
                        pass
                    else:
                        raise RuntimeError("Something's very wrong part 3.")
                    new_bullets.append((y, x, True))
                else: # bullet is over empty block
                    new_bullets.append((y, x, False))
        return reward

    def step(self, action):
        reward = # self._take_action(action)

        done = self.nsteps > 150

        obs = self._next_observation()

        #reward = self._get_reward()

        if action != Action.NOOP:
            # Apply slight penalty for moving
            reward -= 0.005

        return obs.flatten(), reward, done, {}

    def reset(self):
        self._init_obs()
        return self._next_observation().flatten()

    def render(self, mode='human', close=False):
        if mode == 'rgb':
            return self._next_render()
        else:
            return self._next_observation()
