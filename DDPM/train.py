import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from DDPM.DiffusionModel import RewardDiffusionModel
import torch
import torch.nn as nn
import ast

reward_data = pd.read_excel("D:\\tsc_ddqn_prb_1Con_new\\data\\diffusion_data.xlsx")
reward_data = reward_data.iloc[:, 1:3]
# 转换为Tensor
# 确保所有的 nex_state 都是列表，而不是字符串
reward_data['nex_state'] = reward_data['nex_state'].apply(
    lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
print('reward_data[]', reward_data['nex_state'])
# 提取状态和奖励
states = reward_data['nex_state'].tolist()
rewards = reward_data['reward'].tolist()
states_tensor = torch.tensor(states, dtype=torch.float32)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
dataset = TensorDataset(states_tensor, rewards_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型和优化器
model = RewardDiffusionModel(input_dim=len(states[0]), hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练模型并保存最优参数
best_loss = float('inf')


# 训练模型
def train_diffusion_model(model, optimizer, data_loader, num_epochs=10):
    global best_loss
    best_model_path = "D:\\tsc_ddqn_prb_1Con_new\\weights\\best_model_ddpm.pth"
    model.train()
    for epoch in range(num_epochs):
        for states, rewards in data_loader:
            optimizer.zero_grad()
            predicted_rewards = model(states)
            loss = nn.MSELoss()(predicted_rewards, rewards)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            # 保存最优模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model with loss: {best_loss:.4f}')


train_diffusion_model(model, optimizer, data_loader, num_epochs=100)
