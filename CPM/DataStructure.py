import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
#读取state和reward数据
SRData = pd.read_excel(r'D:\tsc_ddqn_prb\data\CPMData_final.xlsx')

#读取对应时间内区域产生的车辆数
ProductData = pd.read_excel(r'D:\tsc_ddqn_prb\data\produced_vehicles.xlsx')

#读取对应时间内区域离开的车辆数
LeaveData = pd.read_excel(r'D:\tsc_ddqn_prb\data\departed_vehicles.xlsx')

print(f"SRdataLength:{len(SRData)},ProductLen:{len(ProductData)},LeaveLen:{len(LeaveData)}")

#向SRData数据集添加y
SRData['y'] = ProductData['Produced Vehicles'] - LeaveData['Departed Vehicles']

# 将列解析为数值数组
def parse_state_column(column):
    return [list(map(float, row.split(', '))) for row in column]

states_rewards = []
for col in ['state_reward_0', 'state_reward_1', 'state_reward_2', 'state_reward_3']:
    states_rewards.append(parse_state_column(SRData[col]))

# 合并所有状态奖励列
states_rewards = np.concatenate(states_rewards, axis=1)  # 合并为一个大的状态矩阵
y = SRData['y'].values


# 滑动窗口处理输入数据
def create_sliding_window(states_rewards, y, window_size):
    inputs = []
    conditions = []
    for i in range(len(states_rewards) - window_size + 1):
        inputs.append(states_rewards[i:i + window_size])  # 滑动窗口
        conditions.append(y[i:i + window_size])  # 取窗口最后的条件
    return np.array(inputs), np.array(conditions)

window_size = 5
input_states_rewards, input_y = create_sliding_window(states_rewards, y, window_size)
# 转换为 Tensor
states_rewards_tensor = torch.tensor(input_states_rewards, dtype=torch.float32)
y_tensor = torch.tensor(input_y, dtype=torch.float32)



# 数据集和 DataLoader
class TrafficDataset(Dataset):
    def __init__(self, states_rewards, conditions):
        self.states = states_rewards
        self.conditions = conditions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.conditions[idx]


dataset = TrafficDataset(states_rewards_tensor, y_tensor)
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
        y = y.unsqueeze(-1) # [batch_size, seq_len, 1]
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
input_dim = input_states_rewards.shape[2]
time_embedding_dim = 32
model = ConditionalDiffusionModel(input_dim=input_dim, condition_dim=1, time_embedding_dim=time_embedding_dim,timesteps=timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()
best_loss = float('inf')
# 训练过程
epochs = 300
for epoch in range(epochs):
    model.train()
    for batch_states_reward, batch_y in data_loader:
        optimizer.zero_grad()
        # 随机时间步
        t = torch.randint(0, timesteps, (batch_states_reward.size(0),), device=batch_states_reward.device)
        # 加噪
        noisy_states, noise = add_noise(batch_states_reward, t, beta_schedule)
        # 预测噪声
        predicted_noise = model(noisy_states, batch_y, t.float())
        # 计算损失
        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'cpm_model.pth')
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
    for _ in range(100):
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
output_path = "cpmgenerated_data.xlsx"
generated_df.to_excel(output_path, index=False)
print(f"生成数据已保存到 '{output_path}'")