import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_excel(r'D:\tsc_ddqn_prb\data\CPMData_final.xlsx')

# 准备状态数据和奖励数据
states = []
rewards = []

#每5行是一个输入
# 前四列是状态，后四列是奖励
for i in range(len(df) - 4):
    state_window = df.iloc[i:i+5, 0:4].values  # 获取状态的窗口
    reward_window = df.iloc[i:i+5, 4:8].values  # 获取奖励的窗口

    # 处理状态数据，将字符串拆分并转换为浮动类型
    state_processed = []
    for state in state_window.flatten():  # 展平状态窗口
        # 假设状态为字符串形式，形如 '0, 0, 0, 0, 0'
        state_values = state.split(',')  # 用逗号分割字符串
        state_processed.extend([float(x) for x in state_values])  # 转换为浮动类型并展开

    # 添加处理后的状态和奖励
    states.append(state_processed)
    avg_rewards = np.mean(reward_window, axis=0)
    rewards.append(avg_rewards)

# 转换为 NumPy 数组，确保数据类型为 float32

states = np.array(states, dtype=np.float32)
rewards = np.array(rewards, dtype=np.float32)
print(states[:1])
# 输出检查转换结果
print(states.shape)
print(rewards.shape)
# 创建 DataFrame 来保存 states 和 rewards
states_df = pd.DataFrame(states)
rewards_df = pd.DataFrame(rewards)

# 创建一个Excel文件
with pd.ExcelWriter('states_rewards.xlsx') as writer:
    # 将 states 和 rewards 数据写入不同的工作表
    states_df.to_excel(writer, sheet_name='States', index=False)
    rewards_df.to_excel(writer, sheet_name='Rewards', index=False)

print("Data saved to 'states_rewards.xlsx' successfully.")



class RewardPredictionDataset(Dataset):
    def __init__(self, states, rewards):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.rewards[idx]




# 创建神经网络模型
class RewardPredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义输入输出维度
input_dim = states.shape[1]  # 展平后的状态维度
output_dim = rewards.shape[1]  # 奖励的数量（4个）

# 初始化模型
model = RewardPredictionModel(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将数据加载到DataLoader中
train_dataset = RewardPredictionDataset(states, rewards)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # 训练模型
# epochs = 100
# for epoch in range(epochs):
#     model.train()  # 设置模型为训练模式
#     running_loss = 0.0
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()  # 清零梯度
#         outputs = model(inputs)  # 前向传播
#         loss = criterion(outputs, targets)  # 计算损失
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新权重
#
#         running_loss += loss.item()  # 累加损失
#
#     # 输出每个epoch的损失
#     print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')
#
# # 保存训练好的模型
# torch.save(model.state_dict(), 'reward_prediction_model.pth')

import torch
import torch.nn as nn

def train_model(model, train_loader, optimizer, criterion, num_epochs=1000, model_save_path='reward_prediction_model.pth'):
    """
    训练模型的函数。

    参数:
    - model: 训练的模型
    - train_loader: 训练数据的 DataLoader
    - optimizer: 优化器
    - criterion: 损失函数
    - num_epochs: 训练的轮次 (默认100)
    - model_save_path: 保存模型的路径 (默认 'reward_prediction_model.pth')
    """
    # 设置最佳损失变量，用于保存最优模型
    best_loss = float('inf')

    # 设置模型为训练模式
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        # 遍历训练数据
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 清零梯度

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()  # 累加损失

        # 计算每个epoch的平均损失
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'保存最优模型，损失: {best_loss:.4f}')

    print('训练完成，最优模型已保存。')

# 调用训练函数示例：
train_model(model, train_loader, optimizer, criterion)




# # 使用模型进行预测
# model.eval()  # 设置模型为评估模式
# with torch.no_grad():  # 禁止梯度计算
#     sample_inputs = torch.tensor(states[:10], dtype=torch.float32)  # 预测前10个样本
#     predictions = model(sample_inputs)
#     print("Predictions on first 10 samples:\n", predictions)
