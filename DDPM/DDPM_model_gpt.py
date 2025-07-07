import torch
import torch.nn as nn


class RewardDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RewardDiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出一个奖励值

    def forward(self, states, t):
        # 正向扩散过程（添加噪声）
        noise = torch.randn_like(states) * (t.float().view(-1, 1) / 10)  # 示例噪声
        noisy_states = states + noise
        return noisy_states

    def reverse(self, noisy_states, t):
        # 逆向扩散过程（去噪并预测奖励）
        x_denoised = torch.relu(self.fc1(noisy_states))
        x_denoised = torch.relu(self.fc2(x_denoised))
        predicted_rewards = self.fc3(x_denoised)  # 预测奖励
        return predicted_rewards


