import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from models.ppo_agent import PPOAgent
from models.features_extractor import MinigridFeaturesExtractor
from training.ppo_trainer import train_ppo
from evaluation.evaluation import evaluate_agent, capture_frames_sb3, count_parameters
from evaluation.visualization import create_animation, plot_parameter_comparison

def custom_ppo_experiment():
    print("Running custom PPO experiment...")
    # Initialize the MiniGrid environment
    env = gym.make('MiniGrid-Empty-5x5-v0')
    env = ImgObsWrapper(env)
    
    # Get the flattened observation shape for correct network sizing
    sample_obs, _ = env.reset()
    obs_space = sample_obs.flatten().shape[0]
    action_space = env.action_space.n
    agent = PPOAgent(obs_space, action_space)
    
    print(f"Observation space flattened shape: {obs_space}")
    print(f"Action space: {action_space}")
    
    # Train the PPO agent
    train_ppo(env, agent, epochs=1000)
    return agent

def sb3_ppo_experiment():
    print("Running Stable Baselines3 PPO experiment...")
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(2e5)
    return model

def evaluate_and_compare(agent, model):
    print("Evaluating and comparing models...")
    # Create test environments
    test_env = gym.make('MiniGrid-Empty-5x5-v0', render_mode="rgb_array")
    test_env = ImgObsWrapper(test_env)
    
    eval_env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    eval_env = ImgObsWrapper(eval_env)
    eval_env = Monitor(eval_env)
    
    # Evaluate custom PPO
    rewards, frames = evaluate_agent(agent, test_env, num_episodes=1, render=True)
    print(f"Average reward per episode (custom): {sum(rewards) / len(rewards) if rewards else 0}")
    
    # Display animation if available
    if frames:
        animation = create_animation(frames[0])
        print("Custom PPO animation created - display in notebook")
    
    # Plot the reward distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(rewards)
    plt.title('Custom PPO Agent Reward Distribution')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.show()
    
    # Evaluate SB3 PPO
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward (SB3): {mean_reward:.2f} +/- {std_reward:.2f}")
    
    sb3_frames = capture_frames_sb3(model, eval_env)
    print("SB3 PPO animation created - display in notebook")
    
    # Compare models
    print("Model architecture comparison:")
    print(f"Custom PPO Agent Architecture:\n{agent}")
    print(f"\nStable Baselines3 PPO Architecture:\n{model.policy}")
    
    # Compare parameter counts
    custom_params = count_parameters(agent)
    sb3_params = count_parameters(model.policy)
    
    print(f"\nParameter Count Comparison:")
    print(f"Custom PPO: {custom_params:,} parameters")
    print(f"Stable Baselines3 PPO: {sb3_params:,} parameters")
    
    # Plot parameter comparison
    plot_parameter_comparison(custom_params, sb3_params)

if __name__ == "__main__":
    agent = custom_ppo_experiment()
    model = sb3_ppo_experiment()
    evaluate_and_compare(agent, model)
