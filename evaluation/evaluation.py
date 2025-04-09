import torch

def evaluate_agent(agent, env, num_episodes=5, render=True, max_steps=100):
    """Evaluate the agent and optionally render frames."""
    all_rewards = []
    all_frames = []
    
    for episode in range(num_episodes):
        episode_rewards = 0
        frames = []
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            if render:
                frames.append(env.render())
                
            with torch.no_grad():
                state = torch.tensor(obs.flatten(), dtype=torch.float32)
                policy, _ = agent(state)
                action = torch.multinomial(policy, 1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards += reward
            step_count += 1
        
        all_rewards.append(episode_rewards)
        if render:
            all_frames.append(frames)
    
    return all_rewards, all_frames

def capture_frames_sb3(model, env, num_steps=100):
    frames = []
    obs, _ = env.reset()
    for _ in range(num_steps):
        frames.append(env.render())
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return frames

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
