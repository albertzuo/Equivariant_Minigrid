import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import e2cnn.nn as enn
import e2cnn.gspaces as gspaces

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class SmallKernelC4EquivariantCNN(BaseFeaturesExtractor):
    """
    C4 Equivariant CNN with small kernels suitable for MiniGrid's small observation spaces.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Define the group structure (C4 = 90-degree rotations)
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        
        # Get input shape
        n_input_channels = observation_space.shape[0]
        
        # Convert standard channel dimension to a group feature field
        self.input_type = enn.FieldType(self.r2_act, n_input_channels * [self.r2_act.trivial_repr])
        
        # Define the sequence of equivariant layers with SMALLER kernels
        self.equivariant_model = nn.Sequential(
            # First equivariant convolution layer - kernel 3x3 (instead of 5x5)
            enn.SequentialModule(
                enn.R2Conv(self.input_type, 
                           enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                           kernel_size=3, 
                           padding=1, 
                           stride=1,
                           frequencies_cutoff=2.0),
                enn.ReLU(enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])),
                enn.PointwiseMaxPool(enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), 2)
            ),
            
            # Second equivariant convolution layer - increased kernel from 2x2 to 3x3 and added frequencies_cutoff
            enn.SequentialModule(
                enn.R2Conv(enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                           enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                           kernel_size=3, 
                           padding=1, 
                           stride=1,
                           frequencies_cutoff=2.0),
                enn.ReLU(enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])),
                enn.PointwiseMaxPool(enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2)  # Added max-pooling
            ),
            
            # Third equivariant convolution layer
            enn.SequentialModule(
                enn.R2Conv(enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                           enn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                           kernel_size=3, 
                           padding=1, 
                           stride=1,
                           frequencies_cutoff=2.0),
                enn.ReLU(enn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]))
            )
        )
        
        sample = torch.randn(1, *observation_space.shape)
        with torch.no_grad():
            geometric_input = enn.GeometricTensor(sample, self.input_type)
            output = self.equivariant_model(geometric_input)
            flattened_size = output.tensor.reshape(1, -1).shape[1]
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        device = next(self.parameters()).device
        x = enn.GeometricTensor(observations.to(device), self.input_type)
        x = self.equivariant_model(x)
        x = x.tensor.reshape(x.tensor.size(0), -1)
        return self.fc(x)