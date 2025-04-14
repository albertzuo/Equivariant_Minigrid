import numpy as np
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

class BaseWrapper(gym.ObservationWrapper):
    def __init__(self, env, full_obs=True):
        if full_obs:
            env = FullyObsWrapper(env)
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    def observation(self, obs):
        return obs['image']

class RandomRotateWrapper(gym.ObservationWrapper):
    def __init__(self, env, subset=[0,1,2,3], full_obs=True):
        if full_obs:
            env = FullyObsWrapper(env)
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
        self.subset = subset
        self.k = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.k = np.random.choice(self.subset)
        return self.observation(obs), info

    def observation(self, obs):
        obs = obs['image']
        img_rot = np.rot90(obs, k=self.k, axes=(0,1)).copy()
        return img_rot

class Rotate90Wrapper(gym.ObservationWrapper):
    def __init__(self, env, full_obs=True):
        if full_obs:
            env = FullyObsWrapper(env)
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    def observation(self, obs):
        obs = obs['image']
        img_rot = np.rot90(obs, k=1, axes=(0,1)).copy()
        return img_rot

class Rotate180Wrapper(gym.ObservationWrapper):
    def __init__(self, env, full_obs=True):
        if full_obs:
            env = FullyObsWrapper(env)
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    def observation(self, obs):
        obs = obs['image']
        img_rot = np.rot90(obs, k=2, axes=(0,1)).copy()
        return img_rot

class Rotate270Wrapper(gym.ObservationWrapper):
    def __init__(self, env, full_obs=True):
        if full_obs:
            env = FullyObsWrapper(env)
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    def observation(self, obs):
        obs = obs['image']
        img_rot = np.rot90(obs, k=3, axes=(0,1)).copy()
        return img_rot
