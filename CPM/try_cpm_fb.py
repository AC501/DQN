import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


file_path = r"D:\tsc_ddqn_prb\data\CPMData_final.xlsx"
data = pd.read_excel(file_path)

# 将列解析为数值数组
def parse_state_column(column):
    return [list(map(float, row.split(', '))) for row in column]

states = []
for col in ['state_0', 'state_1', 'state_2', 'state_3']:
    states.append(parse_state_column(data[col]))

# 合并所有状态列
states = np.concatenate(states, axis=1)  # 合并为一个大的状态矩阵
y = data['y'].values


# 滑动窗口处理输入数据
def create_sliding_window(states, y, window_size):
    inputs = []
    conditions = []
    for i in range(len(states) - window_size + 1):
        inputs.append(states[i:i + window_size])  # 滑动窗口
        conditions.append(y[i + window_size - 1])  # 取窗口最后的条件
    return np.array(inputs), np.array(conditions)

window_size = 5
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
    def __init__(self, input_dim, condition_dim, time_embedding_dim, timesteps):
        super(ConditionalDiffusionModel, self).__init__()

        self.timesteps = timesteps  # 保存时间步数
        self.time_embedding = nn.Linear(1, time_embedding_dim)

        self.linear1 = nn.Linear(input_dim + condition_dim + time_embedding_dim, 128)
        self.linear2 = nn.Linear(128, input_dim)

    def forward(self, x, y, t):
        # 时间步嵌入
        t_embedding = self.time_embedding(t.unsqueeze(-1))  # [batch_size, time_embedding_dim]
        # 扩展 y 和 t_embedding 的维度
        y = y.unsqueeze(-1).unsqueeze(-1).repeat(1, x.size(1), 1)  # [batch_size, seq_len, 1]
        t_embedding = t_embedding.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, seq_len, time_embedding_dim]

        # 拼接特征
        x = torch.cat([x, y, t_embedding], dim=-1)
        # 通过全连接层
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# 定义加噪过程
def add_noise(x, t, beta_schedule):
    beta_t = beta_schedule[t].view(-1, 1, 1)
    noise = torch.randn_like(x)
    noisy_x = torch.sqrt(1 - beta_t) * x + torch.sqrt(beta_t) * noise
    return noisy_x, noise


# 训练参数
timesteps = 100  # 总时间步数
beta_schedule = torch.linspace(1e-5, 0.01, timesteps)  # 噪声强度调度表

# 初始化模型
input_dim = input_states.shape[2]
time_embedding_dim = 32
model = ConditionalDiffusionModel(input_dim=input_dim, condition_dim=1, time_embedding_dim=time_embedding_dim,timesteps=timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# 训练过程
epochs = 100
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
    # 初始化状态，batch_size 为 1
    state = torch.zeros(1, window_size, input_dim)
    # 固定目标条件为 0
    target_condition = 0
    condition = torch.tensor([[target_condition]], dtype=torch.float32).view(1)
    for _ in range(4000):  # 生成 4000 行数据
        # 逐步生成样本数据
        for t in reversed(range(timesteps)):
            # 时间步 t 的张量
            t_tensor = torch.tensor([t], dtype=torch.float32).view(1)  #
            # 预测噪声
            predicted_noise = model(state, condition, t_tensor)
            # 根据反扩散公式计算下一个状态
            beta_t = beta_schedule[t].view(1, 1, 1)  # 扩展为与 state 形状一致
            state = (state - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(1 - beta_t)

        # 将生成的状态加入结果列表，移除 batch 维度
        generated_data.append(state.squeeze(0).numpy())  # 确保数据在 CPU 上进行保存

# 合并生成的数据
generated_data = np.concatenate(generated_data, axis=0)  # 展平成二维数组
generated_df = pd.DataFrame(generated_data, columns=[f"feature_{i}" for i in range(input_dim)])

# 保存为 Excel 文件
output_path = "generated_data.xlsx"
generated_df.to_excel(output_path, index=False)
print(f"生成数据已保存到 '{output_path}'")




