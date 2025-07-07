import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from RewardFunction.FunctionFitting_model import RewardPredictionModel

def rewardfitmain():
    # 读取 Excel 文件
    df = pd.read_excel('data/generated_samples_with_conditions.xlsx')

    # 提取每列的最后一个元素
    # 创建新的列来保存每列的最后一个元素
    df['reward_0'] = df['state_reward_0'].apply(lambda x: int(x.split(',')[-1].strip()))
    df['reward_1'] = df['state_reward_1'].apply(lambda x: int(x.split(',')[-1].strip()))
    df['reward_2'] = df['state_reward_2'].apply(lambda x: int(x.split(',')[-1].strip()))
    df['reward_3'] = df['state_reward_3'].apply(lambda x: int(x.split(',')[-1].strip()))

    # 删除每列的最后一个元素
    df['state_reward_0'] = df['state_reward_0'].apply(lambda x: ','.join(x.split(',')[:-1]).strip())
    df['state_reward_1'] = df['state_reward_1'].apply(lambda x: ','.join(x.split(',')[:-1]).strip())
    df['state_reward_2'] = df['state_reward_2'].apply(lambda x: ','.join(x.split(',')[:-1]).strip())
    df['state_reward_3'] = df['state_reward_3'].apply(lambda x: ','.join(x.split(',')[:-1]).strip())

    # 准备状态数据和奖励数据
    states = []
    rewards = []

    # 每5行是一个输入
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

    # 创建 DataFrame 来保存 states 和 rewards
    states_df = pd.DataFrame(states)
    rewards_df = pd.DataFrame(rewards)

    # 创建一个Excel文件
    with pd.ExcelWriter('states_rewards.xlsx') as writer:
        # 将 states 和 rewards 数据写入不同的工作表
        states_df.to_excel(writer, sheet_name='States', index=False)
        rewards_df.to_excel(writer, sheet_name='Rewards', index=False)
    # 创建数据集
    class RewardPredictionDataset(Dataset):
        def __init__(self, states, rewards):
            self.states = torch.tensor(states, dtype=torch.float32)
            self.rewards = torch.tensor(rewards, dtype=torch.float32)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return self.states[idx], self.rewards[idx]

    # 定义输入输出维度
    input_dim = states.shape[1]  # 展平后的状态维度
    output_dim = rewards.shape[1]  # 奖励的数量（4个）

    # 初始化模型
    model = RewardPredictionModel(input_dim, output_dim)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 将数据加载到 DataLoader 中
    train_dataset = RewardPredictionDataset(states, rewards)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练模型
    epochs = 400
    best_loss = float('inf')  # 初始化最小损失为正无穷
    best_model_state = None  # 保存最优模型的权重
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()  # 累加损失

        avg_loss = running_loss / len(train_loader)
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')
        #
        # # 输出每个 epoch 的损失
        # print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')
        # 检查是否是最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()  # 保存最优模型的权重

        # 保存最优模型
    if best_model_state:
        torch.save(best_model_state, 'RewardFunction/best_reward_prediction_model.pth')
        print("Best model saved!")

    # 保存训练好的模型
    # torch.save(model.state_dict(), 'reward_prediction_model.pth')

    # 如果需要，启用以下代码进行预测
    # model.eval()  # 设置模型为评估模式
    # with torch.no_grad():  # 禁止梯度计算
    #     sample_inputs = torch.tensor(states[:10], dtype=torch.float32)  # 预测前10个样本
    #     predictions = model(sample_inputs)
    #     print("Predictions on first 10 samples:\n", predictions)

if __name__ == "__main__":
    rewardfitmain()
