# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geometric deep learning research project investigating zero-shot rotation generalization in reinforcement learning. The project compares C4-equivariant convolutional neural networks against standard CNNs and data augmentation techniques in a MiniGrid FourRooms environment.

## Development Environment Setup

This project uses Python with a virtual environment setup. Key dependencies are managed through requirements.txt including:
- PyTorch for deep learning models
- stable-baselines3 for PPO reinforcement learning
- escnn for equivariant neural networks  
- gymnasium and minigrid for RL environments
- opencv-python and matplotlib for visualization

To set up:
```bash
pip install -r requirements.txt
```

## Running the Project

### Main Training and Evaluation
The primary research workflow is contained in the Jupyter notebook:
- `minigrid_gdl.ipynb` - Complete training and evaluation pipeline

### Running Tests
Test the environment wrappers:
```bash
python test_wrappers.py
```

## Code Architecture

### Core Components

**Models** (`models/features_extractor.py`):
- `MinigridFeaturesExtractor` - Standard CNN feature extractor for PPO
- `MostlyC4EquivariantCNN` - Partially equivariant CNN with some standard pooling
- `SmallKernelC4EquivariantCNN` - Fully C4-equivariant CNN using escnn library

**Environment Wrappers** (`wrappers.py`):
- `BaseWrapper` - Standard MiniGrid environment wrapper
- `RandomRotateWrapper` - Applies random rotations during training (data augmentation)
- `Rotate90Wrapper`, `Rotate180Wrapper`, `Rotate270Wrapper` - Fixed rotation wrappers for evaluation
- All wrappers support `full_obs=True` for fully observable environments

**Custom Environments** (`envs/custom_fourrooms.py`):
- `FourRoomsEnv` - Base four-rooms navigation environment
- `FourRoomsEnv09`, `FourRoomsEnv11`, `FourRoomsEnv13`, `FourRoomsEnv15`, `FourRoomsEnv17` - Different grid sizes

**Evaluation** (`evaluation/`):
- `evaluation.py` - Agent evaluation functions, video capture, parameter counting
- `visualization.py` - Animation creation utilities

### Training Pipeline

The research workflow follows this pattern:
1. Create vectorized environments with different wrappers (standard, augmented, equivariant)
2. Train PPO agents with different feature extractors
3. Evaluate zero-shot generalization on rotated environments
4. Generate visualizations and comparison videos

### Key Configuration Variables

In the notebook, these variables control the experiment:
- `ENV_NAME` - Environment class to use (e.g., FourRoomsEnv09)
- `TIMESTEPS` - Training duration (default 1e6)
- `NUM_ENVS` - Number of parallel environments (default 32)
- `SEED` - Random seed for reproducibility
- `FULL_OBS` - Whether to use fully observable environments

### Model Checkpoints

Pre-trained models are stored in `model_checkpoints/full/` with naming convention:
- `9x9_standard_agent_10m` - Standard CNN trained for 10M timesteps
- `9x9_eq_agent_10m` - Equivariant CNN trained for 10M timesteps  
- `9x9_aug_agent_10m` - Augmented standard CNN trained for 10M timesteps

### Logging and Monitoring

- Training logs: `./logs/` directory with subdirectories for different agent types
- TensorBoard logs: `./tensorboard_logs/` directory
- Generated videos: `./vids/` directory for agent behavior comparisons

## Key Research Findings

The equivariant agent demonstrates superior zero-shot generalization:
- ~90% success rate across all rotations despite training only on 0° orientation
- Standard CNN fails completely on unseen rotations
- Data augmentation only helps with specifically seen rotations during training