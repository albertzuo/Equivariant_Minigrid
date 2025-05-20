import torch
import numpy as np
import cv2
from pathlib import Path

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

def capture_side_by_side_comparison(models, envs, labels, num_steps=50, rotation=0, output_path=None, scale_factor=2, padding=20, fps=10):
    """
    Capture frames from multiple agents and combine them side by side.
    Args:
        models: List of agent models
        envs: List of environment classes (not instances)
        labels: List of labels for each agent
        num_steps: Maximum number of steps to capture
        rotation: Number of 90-degree clockwise rotations (0, 1, 2, or 3)
        output_path: Optional path to save the video file
        scale_factor: Factor to scale up the frames
        padding: Number of pixels of padding between frames
        fps: Frames per second for the output video (lower = slower)
    """
    if not (len(models) == len(envs) == len(labels)):
        raise ValueError("Number of models, environments, and labels must match")
    
    # Create separate environment instances for each agent
    env_instances = [env() for env in envs]
    
    # Initialize environments
    observations = [env.reset()[0] for env in env_instances]
    frames = []
    
    # Track which environments are done
    envs_done = [False] * len(envs)
    steps_to_complete = [None] * len(envs)
    
    for step in range(num_steps):
        # Capture frame from each agent
        agent_frames = []
        for i, (model, env, obs) in enumerate(zip(models, env_instances, observations)):
            frame = env.render()
            
            # Resize frame
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_NEAREST)
            
            # Create a copy of the frame for rotation
            rotated_frame = frame.copy()
            if rotation > 0:
                rotated_frame = np.rot90(rotated_frame, k=rotation)
            
            # Add timestamp and label overlay
            if envs_done[i]:
                text = f"{labels[i]} Step: {steps_to_complete[i]}"
            else:
                text = f"{labels[i]} Step: {step}"
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.7 * scale_factor  # Scale font size with frame
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
                agent_frames.append(rotated_frame)
            else:
                agent_frames.append(frame)
            
            # Only step if environment is not done
            if not envs_done[i]:
                action, _ = model.predict(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                observations[i] = obs
                
                if terminated:  # Agent reached the goal
                    envs_done[i] = True
                    steps_to_complete[i] = step + 1
        
        # Add padding between frames
        padded_frames = []
        for i, frame in enumerate(agent_frames):
            if i > 0:  # Add padding before each frame except the first
                padding_frame = np.zeros((frame.shape[0], padding, 3), dtype=np.uint8)
                padded_frames.append(padding_frame)
            padded_frames.append(frame)
        
        # Combine frames horizontally
        combined_frame = np.hstack(padded_frames)
        frames.append(combined_frame)
        
        # Stop if all environments are done
        if all(envs_done):
            break
    
    # Save video if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get video properties from the first frame
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write frames to video
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")
    
    return frames

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
