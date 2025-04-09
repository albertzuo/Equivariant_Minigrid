import unittest
import gymnasium as gym
import torch
from minigrid.wrappers import ImgObsWrapper
import sys
import os

# Add parent directory to path to import from sibling modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ppo_agent import PPOAgent
from training.ppo_trainer import train_ppo

class TestPPOImplementation(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('MiniGrid-Empty-5x5-v0')
        self.env = ImgObsWrapper(self.env)
        
        # Get the flattened observation shape for correct network sizing
        sample_obs, _ = self.env.reset()
        obs_space = sample_obs.flatten().shape[0]
        action_space = self.env.action_space.n
        self.agent = PPOAgent(obs_space, action_space)

    def test_policy_update(self):
        initial_policy_params = list(self.agent.policy.parameters())[0].clone()
        train_ppo(self.env, self.agent, epochs=1)
        updated_policy_params = list(self.agent.policy.parameters())[0]
        self.assertFalse(torch.equal(initial_policy_params, updated_policy_params), "Policy parameters did not update.")

    def test_value_update(self):
        initial_value_params = list(self.agent.value.parameters())[0].clone()
        train_ppo(self.env, self.agent, epochs=1)
        updated_value_params = list(self.agent.value.parameters())[0]
        self.assertFalse(torch.equal(initial_value_params, updated_value_params), "Value function parameters did not update.")

    def test_training_stability(self):
        try:
            train_ppo(self.env, self.agent, epochs=5)
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
