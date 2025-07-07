import torch.nn as nn

import torch

# 创建神经网络模型
class RewardPredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

