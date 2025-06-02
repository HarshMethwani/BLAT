import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Proper weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc2.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc3.weight, gain=0.5)
        
        # Initialize biases to small values
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Input normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            # Replace NaN/inf with zeros
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # Scale down extreme values
        if x.abs().max() > 100:
            x = x / (x.abs().max() + 1e-8)
            
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Apply softmax with numerical stability
        action_probs = self.fc3(x)
        # Use log_softmax which is more numerically stable
        log_probs = nn.functional.log_softmax(action_probs, dim=-1)
        action_probs = torch.exp(log_probs)
        
        return action_probs


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)