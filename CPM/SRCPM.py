import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# 数据处理部分（滑动窗口）

def create_sliding_windows(data, state_cols, reward_cols, y_col, window_size=5):
    inputs = []
    conditions = []

    for i in range(len(data) - window_size + 1):
        # 提取每一列的数组，并沿第0轴堆叠（将每个时间步的数据合并为二维数组）
        states = np.stack(data[state_cols].iloc[i: i + window_size].apply(
            lambda row: np.stack(row), axis=1
        ).values).astype(np.float32)

        print('states',states)

        rewards = np.stack(data[reward_cols].iloc[i: i + window_size].apply(
            lambda row: np.array(row, dtype=np.float32), axis=1
        ).values).astype(np.float32)
        print('rewards',rewards)

        y = data[y_col].iloc[i + window_size - 1]
        inputs.append({"states": states, "rewards": rewards})
        conditions.append(y)

    return inputs, conditions


# 加载数据
file_path = r"D:\tsc_ddqn_prb\data\CPMData_final.xlsx"
data = pd.read_excel(file_path)

# 转换state和reward列中的字符串为数组
for col in data.columns:
    if col.startswith("state") or col.startswith("reward"):
        data[col] = data[col].apply(
            lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32) if isinstance(x, str) else np.array(
                [x], dtype=np.float32)
        )

state_columns = [col for col in data.columns if col.startswith("state")]
reward_columns = [col for col in data.columns if col.startswith("reward")]
y_column = "y"

# 生成滑动窗口数据
inputs, conditions = create_sliding_windows(data, state_columns, reward_columns, y_column)


# 自定义数据集
class ConditionDiffusionDataset(Dataset):
    def __init__(self, inputs, conditions):
        self.inputs = inputs
        self.conditions = conditions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        states = torch.tensor(np.array(self.inputs[idx]["states"], dtype=np.float32), dtype=torch.float32)
        rewards = torch.tensor(np.array(self.inputs[idx]["rewards"], dtype=np.float32), dtype=torch.float32)
        condition = torch.tensor(self.conditions[idx], dtype=torch.float32)
        print(f"states_:{states},rewards_:{rewards},condition_:{condition}")
        return states, rewards, condition


dataset = ConditionDiffusionDataset(inputs, conditions)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 条件扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, state_dim, reward_dim, condition_dim, hidden_dim=128, timesteps=1000):
        super(ConditionalDiffusionModel, self).__init__()
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.condition_dim = condition_dim
        self.timesteps = timesteps

        # 噪声预测网络
        self.fc = nn.Sequential(
            nn.Linear(state_dim + reward_dim + condition_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + reward_dim)
        )

    def forward(self, states, rewards, condition, t):
        # 检查输入形状，确保为二维
        if states.ndim > 2:
            states = states.view(states.size(0), -1)  # 展平为 (batch_size, state_dim)
            print('states',states.shape)
        if rewards.ndim > 2:
            rewards = rewards.view(rewards.size(0), -1)  # 展平为 (batch_size, reward_dim)
            print('rewards',rewards.shape)
        print(t.shape())
        t = t.unsqueeze(-1)  # 确保 t 的形状为 (batch_size, 1)

        condition = condition.unsqueeze(-1)  # 确保 condition 的形状为 (batch_size, 1)

        # 拼接状态、奖励、条件和时间步
        x = torch.cat([states, rewards, condition, t], dim=-1)
        return self.fc(x)


# 定义前向扩散过程
def forward_diffusion(x, timesteps, noise=None):
    if noise is None:
        noise = torch.randn_like(x)  # 确保 noise 的形状与 x 一致
    alphas = torch.linspace(0.0001, 0.02, timesteps, device=x.device)  # 自定义时间步长
    alpha_t = alphas[-1]  # 最后时间步的 alpha 值
    noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
    return noisy_x, noise


# 定义损失函数
def diffusion_loss(model, x, condition, timesteps, t, noise):
    noisy_x, target_noise = forward_diffusion(x, timesteps, noise)
    predicted_noise = model(noisy_x, condition, t)
    return nn.MSELoss()(predicted_noise, target_noise)


# 模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = len(state_columns)
reward_dim = len(reward_columns)
condition_dim = 1  # 条件是标量

model = ConditionalDiffusionModel(state_dim, reward_dim, condition_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
timesteps = 1000
for epoch in range(num_epochs):
    epoch_loss = 0
    for states, rewards, condition in dataloader:
        # 确保 states 和 rewards 的形状符合拼接后的要求
        states = states.view(states.size(0), -1).to(device)  # 展平状态
        rewards = rewards.view(rewards.size(0), -1).to(device)  # 展平奖励
        condition = condition.to(device)

        x = torch.cat([states, rewards], dim=-1)  # 拼接后的输入
        noise = torch.randn_like(x).to(device)  # 确保噪声与 x 的形状一致
        t = torch.randint(0, timesteps, (states.size(0),), device=device)  # 随机时间步

        optimizer.zero_grad()
        loss = diffusion_loss(model, x, condition, timesteps, t, noise)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# 条件生成
def generate_samples(model, condition, num_samples, timesteps):
    model.eval()
    with torch.no_grad():
        x = torch.randn((num_samples, state_dim + reward_dim), device=device)
        for t in range(timesteps - 1, -1, -1):
            t_tensor = torch.tensor([t] * num_samples, device=device).unsqueeze(-1)
            pred_noise = model(x, condition.to(device), t_tensor)
            alpha_t = torch.linspace(0.0001, 0.02, timesteps)[t]
            x = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        return x


# 生成示例
condition = torch.tensor([5.0], dtype=torch.float32).to(device)  # 示例条件
generated_data = generate_samples(model, condition, num_samples=10, timesteps=timesteps)
print("Generated Data:", generated_data.cpu().numpy())
