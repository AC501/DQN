import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 读取 Excel 文件
file_path = "your_file.xlsx"  # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path)

# 数据预处理
states = df.iloc[:, :-1].values  # 所有 state 列
y = df.iloc[:, -1].values  # y 列
window_size = 5


# 滑动窗口处理输入数据
def create_sliding_window(states, y, window_size):
    inputs = []
    conditions = []
    for i in range(len(states) - window_size + 1):
        inputs.append(states[i:i + window_size])  # 滑动窗口
        conditions.append(y[i + window_size - 1])  # 取窗口最后的条件
    return np.array(inputs), np.array(conditions)


input_states, input_y = create_sliding_window(states, y, window_size)

# 转换为 Tensor
states_tensor = torch.tensor(input_states, dtype=torch.float32)
y_tensor = torch.tensor(input_y, dtype=torch.float32)


# 数据集和 DataLoader
class TrafficDataset(Dataset):
    def __init__(self, states, conditions):
        self.states = states
        self.conditions = conditions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.conditions[idx]


dataset = TrafficDataset(states_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 定义条件扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, condition_dim, timesteps):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()
        self.timesteps = timesteps

    def forward(self, x, y, t):
        t_embedding = t.unsqueeze(-1)  # 时间步嵌入
        x = torch.cat([x, y.unsqueeze(-1), t_embedding], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# 定义加噪过程
def add_noise(x, t, beta_schedule):
    beta_t = beta_schedule[t].view(-1, 1, 1)
    noise = torch.randn_like(x)
    noisy_x = torch.sqrt(1 - beta_t) * x + torch.sqrt(beta_t) * noise
    return noisy_x, noise


# 训练参数
timesteps = 100  # 总时间步数
beta_schedule = torch.linspace(1e-4, 0.02, timesteps)  # 噪声强度调度表

# 初始化模型
input_dim = input_states.shape[2]
model = ConditionalDiffusionModel(input_dim=input_dim, condition_dim=1, timesteps=timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练过程
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_states, batch_y in data_loader:
        optimizer.zero_grad()

        # 随机时间步
        t = torch.randint(0, timesteps, (batch_states.size(0),), device=batch_states.device)

        # 加噪
        noisy_states, noise = add_noise(batch_states, t, beta_schedule)

        # 预测噪声
        predicted_noise = model(noisy_states, batch_y, t.float())

        # 计算损失
        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# 数据生成过程
model.eval()
generated_data = []
with torch.no_grad():
    # 初始化状态
    state = torch.zeros(1, window_size, input_dim)

    for _ in range(4000):  # 生成 4000 行数据
        condition = torch.tensor([[np.random.choice(y)]], dtype=torch.float32)
        for t in reversed(range(timesteps)):
            t_tensor = torch.tensor([t], dtype=torch.float32)
            predicted_noise = model(state, condition, t_tensor)
            beta_t = beta_schedule[t].view(1, 1, 1)
            state = (state - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(1 - beta_t)
        generated_data.append(state.numpy())

# 保存生成的数据
generated_data = np.concatenate(generated_data, axis=0)
generated_df = pd.DataFrame(generated_data.reshape(-1, input_dim), columns=[f"feature_{i}" for i in range(input_dim)])
generated_df.to_excel("generated_data.xlsx", index=False)
print("生成数据已保存到 'generated_data.xlsx'")
