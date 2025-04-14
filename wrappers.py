import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper
from gymnasium.spaces import Discrete
from minigrid.wrappers import FullyObsWrapper, PositionBonus

class LimitedActionWrapper(ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = Discrete(len(allowed_actions))

    def action(self, act):
        return self.allowed_actions[act]

class _MiniGridWrapperBase(gym.ObservationWrapper):
    def __init__(self, env, full_obs=True, allowed_actions=[0, 1, 2], add_intrinsic=True):
        if full_obs:
            env = FullyObsWrapper(env)
        if allowed_actions:
            env = LimitedActionWrapper(env, allowed_actions=allowed_actions)
        if add_intrinsic:
            env = PositionBonus(env)
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

class BaseWrapper(_MiniGridWrapperBase):
    def __init__(self, env, full_obs=True):
        super().__init__(env, full_obs=full_obs)
    def observation(self, obs):
        return obs['image']

class RandomRotateWrapper(_MiniGridWrapperBase):
    def __init__(self, env, subset=[0,1,2,3], full_obs=True):
        super().__init__(env, full_obs=full_obs)
        self.subset = subset
        self.k = 0
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.k = np.random.choice(self.subset)
        return self.observation(obs), info
    def observation(self, obs):
        obs_img = obs['image'] 
        img_rot = np.rot90(obs_img, k=self.k, axes=(0,1)).copy()
        return img_rot

class Rotate90Wrapper(_MiniGridWrapperBase):
    def __init__(self, env, full_obs=True):
        super().__init__(env, full_obs=full_obs)
    def observation(self, obs):
        obs_img = obs['image'] 
        img_rot = np.rot90(obs_img, k=1, axes=(0,1)).copy()
        return img_rot

class Rotate180Wrapper(_MiniGridWrapperBase):
    def __init__(self, env, full_obs=True):
        super().__init__(env, full_obs=full_obs)
    def observation(self, obs):
        obs_img = obs['image'] 
        img_rot = np.rot90(obs_img, k=2, axes=(0,1)).copy()
        return img_rot

class Rotate270Wrapper(_MiniGridWrapperBase):
    def __init__(self, env, full_obs=True):
        super().__init__(env, full_obs=full_obs)
    def observation(self, obs):
        obs_img = obs['image'] 
        img_rot = np.rot90(obs_img, k=3, axes=(0,1)).copy()
        return img_rot
