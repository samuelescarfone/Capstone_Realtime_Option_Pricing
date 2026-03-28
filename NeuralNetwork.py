import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset

class option_pricing_surrogate(nn.Module):
    def __init__(self, input_size):
            super(option_pricing_surrogate, self).__init__()
            self.network=nn.Sequential(
                  nn.Linear(input_size, 128),
                  nn.ReLU(),
                  nn.Linear(128, 128),
                  nn.ReLU(),
                  nn.Linear(128, 64),
                  nn.ReLU(),
                  nn.Linear(64,1)
                )

def forward(self, x):
      return self.network(x)

num_features = 5
model = option_pricing_surrogate(input_size= num_features)
dummy_inputs = torch.rand(10, num_features)
dummy_predictions = model(dummy_inputs)
