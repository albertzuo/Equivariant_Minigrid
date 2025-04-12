import numpy as np
import gymnasium as gym
from wrappers import Rotate90Wrapper, Rotate180Wrapper, Rotate270Wrapper
from minigrid.wrappers import ImgObsWrapper

def test_rotate90_wrapper_rotation():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    wrapped_env = Rotate90Wrapper(env)
    base_env = ImgObsWrapper(env)
    wrapped_obs, _ = wrapped_env.reset()
    obs, _ = base_env.reset()
    expected_obs = np.rot90(obs, k=1, axes=(0, 1))
    assert np.array_equal(wrapped_obs, expected_obs), "initial observation mismatch"
    for i in range(10):
        action = wrapped_env.action_space.sample()

        _ = wrapped_env.reset()
        wrapped_obs, _, _, _, _ = wrapped_env.step(action)

        _ = base_env.reset()
        obs, _, _, _, _ = base_env.step(action)
        expected_obs = np.rot90(obs, k=1, axes=(0, 1))
        assert np.array_equal(wrapped_obs, expected_obs), "90-degree rotation mismatch"

def test_rotate180_wrapper_rotation():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    wrapped_env = Rotate180Wrapper(env)
    base_env = ImgObsWrapper(env)
    wrapped_obs, _ = wrapped_env.reset()
    obs, _ = base_env.reset()
    expected_obs = np.rot90(obs, k=2, axes=(0, 1))
    assert np.array_equal(wrapped_obs, expected_obs), "initial observation mismatch"
    for i in range(10):
        action = wrapped_env.action_space.sample()

        _ = wrapped_env.reset()
        wrapped_obs, _, _, _, _ = wrapped_env.step(action)

        _ = base_env.reset()
        obs, _, _, _, _ = base_env.step(action)
        expected_obs = np.rot90(obs, k=2, axes=(0, 1))
        assert np.array_equal(wrapped_obs, expected_obs), "180-degree rotation mismatch"

def test_rotate270_wrapper_rotation():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    wrapped_env = Rotate270Wrapper(env)
    base_env = ImgObsWrapper(env)
    wrapped_obs, _ = wrapped_env.reset()
    obs, _ = base_env.reset()
    expected_obs = np.rot90(obs, k=3, axes=(0, 1))
    assert np.array_equal(wrapped_obs, expected_obs), "initial observation mismatch"
    for i in range(10):
        action = wrapped_env.action_space.sample()

        _ = wrapped_env.reset()
        wrapped_obs, _, _, _, _ = wrapped_env.step(action)

        _ = base_env.reset()
        obs, _, _, _, _ = base_env.step(action)
        expected_obs = np.rot90(obs, k=3, axes=(0, 1))
        assert np.array_equal(wrapped_obs, expected_obs), "270-degree rotation mismatch"

if __name__ == "__main__":
    test_rotate90_wrapper_rotation()
    test_rotate180_wrapper_rotation()
    test_rotate270_wrapper_rotation()
    print("All rotation tests passed!")
