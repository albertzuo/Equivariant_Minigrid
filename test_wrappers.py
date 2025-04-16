import numpy as np
import gymnasium as gym
from wrappers import Rotate90Wrapper, Rotate180Wrapper, Rotate270Wrapper, BaseWrapper, RandomRotateWrapper, OneHotObsWrapper, TransposeImageWrapper
from minigrid.wrappers import FullyObsWrapper
from envs.custom_fourrooms import FourRoomsEnv17, FourRoomsEnv15, FourRoomsEnv13

ENV_NAME = "MiniGrid-FourRooms-v0"

def make_starting_env(env_name=ENV_NAME):
    if env_name == "MiniGrid-FourRooms-v0":
        env = gym.make(env_name)
    else:
        env = env_name()
    return env

def test_wrapper_base(wrapper_class, k=0, debug=False):
    seed = 42
    env = make_starting_env()
    
    base_env = FullyObsWrapper(env)
    # base_env = OneHotObsWrapper(base_env)
    # base_env = TransposeImageWrapper(base_env)
    obs, _ = base_env.reset(seed=seed)
    obs = obs['image']
    expected_obs = np.rot90(obs, k=k, axes=(0, 1))

    env2 = make_starting_env()
    wrapped_env = wrapper_class(env2)
    wrapped_obs, _ = wrapped_env.reset(seed=seed)
    if debug:
        print(expected_obs.shape)
        print('--'*20)
        print(wrapped_obs.shape)
        # print(wrapped_obs, expected_obs)
    assert np.allclose(wrapped_obs, expected_obs), f"{wrapper_class.__name__} initial observation mismatch"
    for i in range(10):
        action = wrapped_env.action_space.sample()

        _ = base_env.reset()
        obs, _, _, _, _ = base_env.step(action)
        obs = obs['image']
        expected_obs = np.rot90(obs, k=k, axes=(0, 1))

        _ = wrapped_env.reset()
        wrapped_obs, _, _, _, _ = wrapped_env.step(action)
        
        if debug:
            print(expected_obs)
            print('--'*20)
            print(wrapped_obs)
        assert np.allclose(wrapped_obs, expected_obs), f"{wrapper_class.__name__} rotation mismatch"

def test_rotate90_wrapper_rotation():
    test_wrapper_base(Rotate90Wrapper, k=1)

def test_rotate180_wrapper_rotation():
    test_wrapper_base(Rotate180Wrapper, k=2)

def test_rotate270_wrapper_rotation():
    test_wrapper_base(Rotate270Wrapper, k=3)

def test_observation_normalization(wrapper_class, **kwargs):
    env = gym.make("MiniGrid-Empty-5x5-v0")
    wrapped_env = wrapper_class(env, **kwargs) if kwargs else wrapper_class(env)
    obs, _ = wrapped_env.reset()
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"{wrapper_class.__name__} failed normalization test"

def test_wrappers_normalization():
    test_observation_normalization(BaseWrapper)
    test_observation_normalization(RandomRotateWrapper, subset=[0, 1, 2, 3])
    test_observation_normalization(Rotate90Wrapper)
    test_observation_normalization(Rotate180Wrapper)
    test_observation_normalization(Rotate270Wrapper)

if __name__ == "__main__":
    test_rotate90_wrapper_rotation()
    test_rotate180_wrapper_rotation()
    test_rotate270_wrapper_rotation()
    # test_wrappers_normalization()
    print("All rotation tests passed!")
    print("All tests passed!")
