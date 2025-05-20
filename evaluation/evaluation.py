import torch
import numpy as np
import cv2

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

def capture_frames_sb3(model, env, num_steps=100, rotation=0):
    """
    Capture frames from the environment with optional rotation and timestamp overlay.
    Args:
        model: The agent model
        env: The environment
        num_steps: Maximum number of steps to capture
        rotation: Number of 90-degree clockwise rotations (0, 1, 2, or 3)
    """
    frames = []
    obs, _ = env.reset()
    for step in range(num_steps):
        frame = env.render()
        
        # Create a copy of the frame for rotation
        rotated_frame = frame.copy()
        if rotation > 0:
            rotated_frame = np.rot90(rotated_frame, k=rotation)
        
        # Add timestamp overlay to the original frame
        text = f"Step: {step}"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.7
        font_thickness = 1
        text_color = (255, 255, 255)  # White color
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # Add the text to the original frame
        cv2.putText(frame, text, (10, 10 + text_size[1]), font, font_scale, text_color, font_thickness)
        
        # Combine the rotated frame with the text overlay
        if rotation > 0:
            # Create a mask for the text region
            mask = np.zeros_like(rotated_frame)
            text_region = frame[0:text_size[1] + 20, 0:text_size[0] + 20]
            mask[0:text_size[1] + 20, 0:text_size[0] + 20] = text_region
            
            # Blend the text region with the rotated frame
            rotated_frame = np.where(mask != 0, mask, rotated_frame)
            frames.append(rotated_frame)
        else:
            frames.append(frame)
        
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return frames

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
