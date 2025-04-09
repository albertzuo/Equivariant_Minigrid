import torch
from torch import nn

class PPOAgent(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PPOAgent, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.policy(x), self.value(x)
