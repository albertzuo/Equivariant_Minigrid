import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, spaces
from gymnasium.spaces import Discrete
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

class LimitedActionWrapper(ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = Discrete(len(allowed_actions))

    def action(self, act):
        return self.allowed_actions[act]

class OneHotObsWrapper(ObservationWrapper):
    def __init__(self, env, tile_size=8):
        """A wrapper that makes the image observation a one-hot encoding of a partially observable agent view.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space["image"].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX) + 1
        # The last bit is for the agent's position

        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}


class TransposeImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        orig_shape = self.observation_space['image'].shape  # (H, W, C)
        h, w, c = orig_shape
        self.observation_space = gym.spaces.Dict({
            **self.observation_space.spaces,
            "image": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(c, h, w),
                dtype=np.float32
            )
        })

    def observation(self, obs):
        obs = {**obs}
        obs["image"] = np.transpose(obs["image"], (2, 0, 1))  # (H, W, C) → (C, H, W)
        return obs

class _MiniGridWrapperBase(gym.ObservationWrapper):
    def __init__(self, env, full_obs=True, allowed_actions=[0, 1, 2]):
        if full_obs:
            env = FullyObsWrapper(env)
            # env = OneHotObsWrapper(env)
            # env = TransposeImageWrapper(env)
        if allowed_actions:
            env = LimitedActionWrapper(env, allowed_actions=allowed_actions)
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
